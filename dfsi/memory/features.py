"""Memory, part 2 — turning the logbook into clues (PySpark).

Reads the raw ANL-shaped 10-second windows and produces the DFSI feature table
plus the 120-second onset label.

Features
--------
traffic_growth       relative change in pps and Mbps over 10s and 60s windows
baseline_deviation   (current rate - median for the same hour-of-week) / MAD,
                     computed per (range, traffic_type)
source_profile       country-mix concentration + the account's prior attack
                     count (no raw IPs anywhere in the pipeline)
blocked_share_delta  change in the blocked fraction of traffic
anomaly_persistence   number of consecutive abnormal windows so far

Label
-----
Per traffic-type row: 1 if Shared State declares an attack on the range in
(t, t+120s] *and this traffic type is over its start threshold at that
declaration*, else 0. The declaration itself is range-level; the label says
which vectors carry it, so a quiet vector's row is never asked to predict an
onset its own features cannot see. The serving layer combines the per-type
scores into one range-level risk.

Anti-leakage rules enforced here
--------------------------------
* every feature is built from a window frame that ends at the current row
  (`rowsBetween(..., 0)`) — no future-derived value can enter a feature;
* the label is the only forward-looking column, and it is dropped from the
  feature vector;
* an episode's baseline comes only from the same hour of week in *earlier*
  weeks (`compute_trailing_baselines`), so no episode — and no test week —
  ever contributes to its own baseline. The first week is warm-up: it feeds
  baselines and never reaches the feature table;
* `scenario_id` and `week_index` are carried for splitting and are never model
  inputs.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from ..config import (
    ANOMALY_DEVIATION_THRESHOLD,
    BASELINE_LOOKBACK_WEEKS,
    FEATURE_VERSION,
    HORIZON_WINDOWS,
    PERSISTENCE_LOOKBACK_WINDOWS,
    WINDOW_SECONDS,
    SharedStateThresholds,
)

RAW_SCHEMA = StructType(
    [
        StructField("scenario_id", StringType(), False),
        StructField("scenario_type", StringType(), False),
        StructField("scenario_set_version", StringType(), False),
        StructField("window_start_epoch", LongType(), False),
        StructField("range_id", LongType(), False),
        StructField("account_id", LongType(), False),
        StructField("range_type", StringType(), False),
        StructField("traffic_type", StringType(), False),
        StructField("total_pps", DoubleType(), False),
        StructField("total_mbps", DoubleType(), False),
        StructField("blocked_pps", DoubleType(), False),
        StructField("blocked_mbps", DoubleType(), False),
        StructField("policy_pps", DoubleType(), False),
        StructField("policy_mbps", DoubleType(), False),
        StructField("country_hhi", DoubleType(), False),
        StructField("account_attack_history", IntegerType(), False),
        StructField("is_attack_vector", BooleanType(), False),
        StructField("week_index", IntegerType(), False),
    ]
)

#: Join key between raw windows and their baselines. `week_index` is the week
#: the baseline is *for*; it is built from strictly earlier weeks.
BASELINE_KEYS: tuple[str, ...] = ("range_id", "traffic_type", "hour_of_week", "week_index")

#: Exactly the columns handed to the forecaster. Keep this list authoritative —
#: `scenario_id`, `scenario_type` and the label must never appear in it.
FEATURE_COLUMNS: tuple[str, ...] = (
    "growth_pps_10s",
    "growth_pps_60s",
    "growth_mbps_10s",
    "growth_mbps_60s",
    "baseline_deviation_pps",
    "baseline_deviation_mbps",
    "baseline_deviation_pps_max_hour",
    "country_hhi",
    "country_hhi_delta_60s",
    "account_attack_history",
    "blocked_share",
    "blocked_share_delta_10s",
    "blocked_share_delta_60s",
    "anomaly_persistence",
    "policy_utilisation",
    "policy_utilisation_mbps",
)

LABEL_COLUMN = "label_onset_120s"
GROUP_COLUMN = "scenario_id"

_WINDOWS_PER_MINUTE = 60 // WINDOW_SECONDS  # 6


def build_spark(
    app_name: str = "dfsi-memory", shuffle_partitions: int | None = None
) -> SparkSession:
    """Local SparkSession sized for a laptop demo.

    Pins the worker interpreter to the one running this process. Without it,
    Spark launches workers with whatever `python3` is first on PATH, and a
    driver/worker minor-version mismatch aborts the job.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # In local mode the driver is the only executor. Spark's 1g default
        # runs out of heap on the four-week logbook.
        .config("spark.driver.memory", os.getenv("DFSI_SPARK_DRIVER_MEMORY", "12g"))
        # The window features hold a whole partition per task. With 42 ranges
        # the four-week logbook is ~22M rows; at 8 partitions each task held an
        # eighth of it and the heap ran out writing the feature table. Many
        # smaller partitions keep the per-task working set bounded.
        .config(
            "spark.sql.shuffle.partitions",
            str(shuffle_partitions or int(os.getenv("DFSI_SPARK_SHUFFLE_PARTITIONS", "96"))),
        )
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def _safe_ratio(numerator, denominator):
    """numerator / denominator, yielding 0.0 when the denominator vanishes."""
    return F.when(F.abs(denominator) < F.lit(1e-9), F.lit(0.0)).otherwise(numerator / denominator)


def with_time_keys(raw: DataFrame) -> DataFrame:
    """Add the hour-of-week key used for baselines."""
    ts = F.timestamp_seconds(F.col("window_start_epoch"))
    return raw.withColumn(
        # dayofweek is 1=Sunday..7=Saturday; shift to 0..6 then combine.
        "hour_of_week",
        ((F.dayofweek(ts) - F.lit(1)) * F.lit(24) + F.hour(ts)).cast("int"),
    )


def compute_baselines(raw_with_keys: DataFrame) -> DataFrame:
    """Median and MAD per (range_id, traffic_type, hour_of_week).

    Aggregates whatever windows it is given; `compute_trailing_baselines`
    decides which windows those are. MAD is scaled by 1.4826 so that for
    normally distributed traffic the resulting deviation reads like a z-score.

    Only *quiet* windows feed the baseline: windows where blocking already
    exceeds the policy-relative start threshold describe an attack, not normal
    behaviour, and would inflate the median a forecast is measured against.
    """
    quiet = raw_with_keys.filter(F.col("blocked_pps") <= F.col("policy_pps") * F.lit(0.05))
    keys = ["range_id", "traffic_type", "hour_of_week"]

    medians = quiet.groupBy(*keys).agg(
        F.expr("percentile_approx(total_pps, 0.5, 1000)").alias("median_pps"),
        F.expr("percentile_approx(total_mbps, 0.5, 1000)").alias("median_mbps"),
        F.count(F.lit(1)).alias("baseline_sample_count"),
        # Provenance, so a test can prove which weeks fed each baseline.
        F.min("week_index").alias("source_week_min"),
        F.max("week_index").alias("source_week_max"),
    )

    deviations = (
        quiet.join(F.broadcast(medians), on=keys, how="inner")
        .withColumn("abs_dev_pps", F.abs(F.col("total_pps") - F.col("median_pps")))
        .withColumn("abs_dev_mbps", F.abs(F.col("total_mbps") - F.col("median_mbps")))
        .groupBy(*keys)
        .agg(
            F.expr("percentile_approx(abs_dev_pps, 0.5, 1000)").alias("mad_pps_raw"),
            F.expr("percentile_approx(abs_dev_mbps, 0.5, 1000)").alias("mad_mbps_raw"),
        )
    )

    return (
        medians.join(deviations, on=keys, how="inner")
        # Floor the MAD so a near-constant baseline cannot explode the ratio.
        .withColumn(
            "mad_pps",
            F.greatest(
                F.col("mad_pps_raw") * F.lit(1.4826), F.col("median_pps") * F.lit(0.02), F.lit(1.0)
            ),
        )
        .withColumn(
            "mad_mbps",
            F.greatest(
                F.col("mad_mbps_raw") * F.lit(1.4826),
                F.col("median_mbps") * F.lit(0.02),
                F.lit(0.01),
            ),
        )
        .drop("mad_pps_raw", "mad_mbps_raw")
    )


def compute_trailing_baselines(
    raw_with_keys: DataFrame, lookback_weeks: int = BASELINE_LOOKBACK_WEEKS
) -> DataFrame:
    """Baselines for each week, from up to `lookback_weeks` *earlier* weeks.

    This is how a production baseline works — it can only have seen the past —
    and it is what stops an episode from being measured against itself. A
    single-week logbook has exactly one episode per (range, hour of week), so
    an all-data baseline would be that episode's own median, future included.

    The returned `week_index` is the week the baseline is *for*. The earliest
    week has no past and gets no baseline: it is warm-up.
    """
    weeks = sorted(
        int(row["week_index"]) for row in raw_with_keys.select("week_index").distinct().collect()
    )
    if len(weeks) < 2:
        raise ValueError(
            f"need at least two weeks of windows to build past-only baselines; got weeks {weeks}. "
            "Generate more days (`--days 28`)."
        )
    parts: list[DataFrame] = []
    for week in weeks[1:]:
        source = raw_with_keys.filter(
            (F.col("week_index") < F.lit(week))
            & (F.col("week_index") >= F.lit(week - lookback_weeks))
        )
        parts.append(compute_baselines(source).withColumn("week_index", F.lit(week).cast("int")))
    result = parts[0]
    for part in parts[1:]:
        result = result.unionByName(part)
    return result


def build_features(
    raw: DataFrame,
    baselines: DataFrame,
    thresholds: SharedStateThresholds,
) -> DataFrame:
    """Join baselines, compute every feature, then attach the 120s label.

    Only weeks that have a past-only baseline are featurised; the warm-up week
    is dropped here, before any window function runs.
    """
    keyed = with_time_keys(raw)
    covered_weeks = baselines.select("week_index").distinct()
    keyed = keyed.join(F.broadcast(covered_weeks), on="week_index", how="inner")
    joined = keyed.join(
        F.broadcast(baselines.drop("source_week_min")),
        on=list(BASELINE_KEYS),
        how="left",
    )

    # ---------------- per-series trailing windows (never look forward) -----
    series = Window.partitionBy("scenario_id", "range_id", "traffic_type").orderBy(
        "window_start_epoch"
    )
    trail_60s = series.rowsBetween(-_WINDOWS_PER_MINUTE, 0)
    trail_persist = series.rowsBetween(-PERSISTENCE_LOOKBACK_WINDOWS, 0)

    prev_pps_10s = F.lag("total_pps", 1).over(series)
    prev_pps_60s = F.lag("total_pps", _WINDOWS_PER_MINUTE).over(series)
    prev_mbps_10s = F.lag("total_mbps", 1).over(series)
    prev_mbps_60s = F.lag("total_mbps", _WINDOWS_PER_MINUTE).over(series)

    feats = (
        joined
        # traffic_growth — relative change over 10s and 60s.
        .withColumn(
            "growth_pps_10s",
            F.coalesce(_safe_ratio(F.col("total_pps") - prev_pps_10s, prev_pps_10s), F.lit(0.0)),
        )
        .withColumn(
            "growth_pps_60s",
            F.coalesce(_safe_ratio(F.col("total_pps") - prev_pps_60s, prev_pps_60s), F.lit(0.0)),
        )
        .withColumn(
            "growth_mbps_10s",
            F.coalesce(_safe_ratio(F.col("total_mbps") - prev_mbps_10s, prev_mbps_10s), F.lit(0.0)),
        )
        .withColumn(
            "growth_mbps_60s",
            F.coalesce(_safe_ratio(F.col("total_mbps") - prev_mbps_60s, prev_mbps_60s), F.lit(0.0)),
        )
        # baseline_deviation — robust z-score against the same hour of week.
        .withColumn(
            "baseline_deviation_pps",
            F.coalesce(
                _safe_ratio(F.col("total_pps") - F.col("median_pps"), F.col("mad_pps")),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "baseline_deviation_mbps",
            F.coalesce(
                _safe_ratio(F.col("total_mbps") - F.col("median_mbps"), F.col("mad_mbps")),
                F.lit(0.0),
            ),
        )
        # blocked_share_delta — how fast the blocked fraction is moving.
        .withColumn(
            "blocked_share",
            F.coalesce(_safe_ratio(F.col("blocked_pps"), F.col("total_pps")), F.lit(0.0)),
        )
        .withColumn("policy_utilisation", _safe_ratio(F.col("total_pps"), F.col("policy_pps")))
        .withColumn(
            "policy_utilisation_mbps",
            _safe_ratio(F.col("total_mbps"), F.col("policy_mbps")),
        )
    )

    feats = (
        feats.withColumn(
            "blocked_share_delta_10s",
            F.col("blocked_share") - F.coalesce(F.lag("blocked_share", 1).over(series), F.lit(0.0)),
        )
        .withColumn(
            "blocked_share_delta_60s",
            F.col("blocked_share")
            - F.coalesce(F.lag("blocked_share", _WINDOWS_PER_MINUTE).over(series), F.lit(0.0)),
        )
        .withColumn(
            "country_hhi_delta_60s",
            F.col("country_hhi")
            - F.coalesce(
                F.lag("country_hhi", _WINDOWS_PER_MINUTE).over(series), F.col("country_hhi")
            ),
        )
        # anomaly_persistence — consecutive abnormal windows within a trailing
        # lookback. A rolling sum of the abnormal flag over the trailing frame
        # is the leak-free form: it never reads a row after the current one.
        .withColumn(
            "is_abnormal",
            (F.abs(F.col("baseline_deviation_pps")) > F.lit(ANOMALY_DEVIATION_THRESHOLD)).cast(
                "int"
            ),
        )
        .withColumn("anomaly_persistence", F.sum("is_abnormal").over(trail_persist))
        .withColumn("blocked_share_mean_60s", F.avg("blocked_share").over(trail_60s))
    )

    # baseline_deviation across *all* traffic types of the range at this
    # instant — the strongest single vector often leads the others.
    per_instant = Window.partitionBy("scenario_id", "range_id", "window_start_epoch")
    feats = feats.withColumn(
        "baseline_deviation_pps_max_hour",
        F.max("baseline_deviation_pps").over(per_instant),
    )

    return add_onset_label(feats, thresholds)


def add_onset_label(feats: DataFrame, thresholds: SharedStateThresholds) -> DataFrame:
    """Attach `label_onset_120s`: does qualifying blocking *begin* soon?

    A window is "over threshold" when blocked traffic exceeds the policy-
    relative start threshold **and** clears the absolute start floor — exactly
    the conjunction Shared State applies before it starts counting duration.

    The duration and horizon arithmetic is done on a frame holding exactly one
    row per (scenario, range, instant). `feats` has one row per traffic type,
    so a `rowsBetween` frame over it would count rows rather than windows —
    which silently divides every duration by the number of traffic types.
    """
    over = (
        (F.col("blocked_pps") >= F.col("policy_pps") * F.lit(thresholds.start_threshold_ratio))
        & (F.col("blocked_pps") >= F.lit(thresholds.start_floor_kpps * 1_000.0))
        & (F.col("blocked_mbps") >= F.lit(thresholds.start_floor_mbps))
    )
    ranged = feats.withColumn("over_threshold", over.cast("int"))

    # One row per instant: the range is over threshold if any traffic type is.
    per_instant = ranged.groupBy("scenario_id", "range_id", "window_start_epoch").agg(
        F.max("over_threshold").alias("range_over_threshold")
    )

    instant = Window.partitionBy("scenario_id", "range_id").orderBy("window_start_epoch")

    # Sustained blocking: the range has been over threshold for the full start
    # duration. This is the moment the bell would actually ring.
    sustain = instant.rowsBetween(-(thresholds.start_duration_windows - 1), 0)
    per_instant = per_instant.withColumn(
        "sustained_windows", F.sum("range_over_threshold").over(sustain)
    ).withColumn(
        "declared",
        (F.col("sustained_windows") >= F.lit(thresholds.start_duration_windows)).cast("int"),
    )

    # Onset = the first declared window of a contiguous declared run.
    prev_declared = F.coalesce(F.lag("declared", 1).over(instant), F.lit(0))
    per_instant = per_instant.withColumn(
        "is_onset", ((F.col("declared") == 1) & (prev_declared == 0)).cast("int")
    )

    labelled = ranged.join(
        per_instant, on=["scenario_id", "range_id", "window_start_epoch"], how="left"
    )

    # Forward-looking by design, per traffic type: 1 if the range declares in
    # (t, t+120s] and *this* type is over threshold at that declaration. The
    # frame holds one row per window per type, so `rowsBetween` counts windows.
    carries_onset = ((F.col("is_onset") == 1) & (F.col("over_threshold") == 1)).cast("int")
    per_type = Window.partitionBy("scenario_id", "range_id", "traffic_type").orderBy(
        "window_start_epoch"
    )
    horizon = per_type.rowsBetween(1, HORIZON_WINDOWS)
    labelled = labelled.withColumn("_carries_onset", carries_onset).withColumn(
        LABEL_COLUMN,
        (F.coalesce(F.sum("_carries_onset").over(horizon), F.lit(0)) > 0).cast("int"),
    )
    return labelled.drop("_carries_onset").withColumn("feature_version", F.lit(FEATURE_VERSION))
