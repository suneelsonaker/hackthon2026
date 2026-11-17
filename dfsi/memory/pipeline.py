"""Memory, part 3 — the pipeline driver.

    synthetic episodes -> raw windows (parquet, one file per week)
                       -> past-only hour-of-week baselines (parquet)
                       -> feature table + 120s label (parquet; warm-up week dropped)
                       -> per-scenario replay files (json, for the UI; test week only)

Run with:  python -m dfsi.memory.pipeline --days 28
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from ..config import (
    CALIBRATION_WEEKS,
    PATHS,
    SCENARIO_SET_VERSION,
    TEST_WEEKS,
    thresholds_for,
)
from .features import (
    LABEL_COLUMN,
    RAW_SCHEMA,
    build_features,
    build_spark,
    compute_trailing_baselines,
    with_time_keys,
)
from .scenarios import (
    ATTACK_SCENARIOS,
    DEFAULT_RANGES,
    EpisodePlan,
    generate_episode_columns,
    plan_week,
)

#: Warm-up (baseline only) + at least one fit week + calibration + test.
MIN_WEEKS_FOR_TIME_SPLIT = 2 + CALIBRATION_WEEKS + TEST_WEEKS

log = logging.getLogger("dfsi.memory")

#: Columns a replay row needs: enough to re-score the window live in the API
#: and to draw the timeline, nothing more.
REPLAY_COLUMNS = (
    "scenario_id",
    "scenario_type",
    "window_start_epoch",
    "range_id",
    "account_id",
    "range_type",
    "traffic_type",
    "total_pps",
    "total_mbps",
    "blocked_pps",
    "blocked_mbps",
    "policy_pps",
    "policy_mbps",
    "country_hhi",
    "account_attack_history",
    "growth_pps_10s",
    "growth_pps_60s",
    "growth_mbps_10s",
    "growth_mbps_60s",
    "baseline_deviation_pps",
    "baseline_deviation_mbps",
    "baseline_deviation_pps_max_hour",
    "country_hhi_delta_60s",
    "blocked_share",
    "blocked_share_delta_10s",
    "blocked_share_delta_60s",
    "anomaly_persistence",
    "policy_utilisation",
    "policy_utilisation_mbps",
    "range_over_threshold",
    "declared",
    "is_onset",
    "label_onset_120s",
)


def run(days: int = 28, environment: str | None = None, replay_per_type: int | None = None) -> dict:
    """Build the whole Memory layer. Returns a summary dict."""
    thresholds = thresholds_for(environment)
    for directory in (PATHS.raw_windows, PATHS.baselines, PATHS.features, PATHS.replay):
        directory.parent.mkdir(parents=True, exist_ok=True)

    plans = plan_week(DEFAULT_RANGES, days=days)
    weeks = sorted({plan.week_index for plan in plans})
    log.info("planned %d episodes over %d days (%d weeks)", len(plans), days, len(weeks))
    if len(weeks) < MIN_WEEKS_FOR_TIME_SPLIT:
        log.warning(
            "only %d weeks planned; the time split needs %d (warm-up, fit, calibration, test). "
            "Train with `--split scenario` or build more days.",
            len(weeks),
            MIN_WEEKS_FOR_TIME_SPLIT,
        )

    window_count = _write_raw_windows(plans)
    log.info("generated %d raw windows", window_count)

    spark = build_spark()
    try:
        raw = spark.read.parquet(str(PATHS.raw_windows))

        baselines = compute_trailing_baselines(with_time_keys(raw)).cache()
        baselines.write.mode("overwrite").parquet(str(PATHS.baselines))

        features = build_features(raw, baselines, thresholds).cache()
        features.write.mode("overwrite").parquet(str(PATHS.features))

        label_counts = {
            int(r[LABEL_COLUMN]): int(r["count"])
            for r in features.groupBy(LABEL_COLUMN).count().collect()
        }
        by_type = {
            r["scenario_type"]: int(r["count"])
            for r in features.groupBy("scenario_type").count().collect()
        }
        featurised_weeks = sorted(
            int(r["week_index"]) for r in features.select("week_index").distinct().collect()
        )
        test_weeks = featurised_weeks[-TEST_WEEKS:]
        _write_replays(features, replay_per_type, test_weeks)
        summary = {
            "scenario_set_version": SCENARIO_SET_VERSION,
            "environment": thresholds.environment,
            "days": days,
            "episodes": len(plans),
            "episodes_by_type": dict(sorted(Counter(p.scenario_type for p in plans).items())),
            "windows": window_count,
            "weeks": weeks,
            "featurised_weeks": featurised_weeks,
            "replay_weeks": test_weeks,
            "label_counts": label_counts,
            "rows_by_scenario_type": by_type,
        }
    finally:
        spark.stop()

    (PATHS.data / "memory_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_raw_windows(plans: list[EpisodePlan]) -> int:
    """Generate the logbook straight to parquet, one file per week.

    Columnar generation keeps a multi-week logbook out of Python dicts; Spark
    reads the directory back with the same schema it would have inferred.
    """
    target = PATHS.raw_windows
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    columns = [field.name for field in RAW_SCHEMA.fields]
    by_week: dict[int, list[EpisodePlan]] = defaultdict(list)
    for plan in plans:
        by_week[plan.week_index].append(plan)

    total = 0
    for week, week_plans in sorted(by_week.items()):
        frame = pd.concat(
            [pd.DataFrame(generate_episode_columns(plan)) for plan in week_plans],
            ignore_index=True,
        )[columns]
        frame.to_parquet(target / f"week-{week:02d}.parquet", index=False)
        total += len(frame)
    return total


def choose_replays(rows: list, per_type: int | None, prefer_onset: bool) -> list:
    """Pick up to `per_type` scenarios of one family, spread across accounts, then ranges.

    `rows` need `scenario_id`, `range_id`, `account_id` and `has_onset`. With
    `per_type=None`, one per account. Attack families put scenarios whose
    bell actually rings first. Deterministic: ties break on `scenario_id`.
    """
    limit = per_type if per_type is not None else len({r["account_id"] for r in rows})
    ordered = sorted(rows, key=lambda r: (prefer_onset and not r["has_onset"], r["scenario_id"]))
    bucket: list = []
    for spread in ("account_id", "range_id"):  # one per account, then one per range
        seen = {r[spread] for r in bucket}
        for row in ordered:
            if len(bucket) < limit and row[spread] not in seen and row not in bucket:
                bucket.append(row)
                seen.add(row[spread])
    for row in ordered:  # then fill
        if len(bucket) < limit and row not in bucket:
            bucket.append(row)
    return bucket


def _write_replays(features, per_type: int | None, weeks: list[int]) -> None:
    """Export a handful of complete scenarios as JSON for the UI replay.

    One replay = every window of one episode for one range, ordered in time,
    so the dashboard can stream it without Spark in the loop.

    Replays come from the held-out test week(s) only, so the demo never shows
    the model a scenario it was fitted or calibrated on. Within a family,
    attack replays that actually ring the bell come first, and accounts, then
    ranges, are spread before any repeats — so the dashboard can offer every
    account. `per_type=None` (the default) means one replay per account.
    """
    PATHS.replay.mkdir(parents=True, exist_ok=True)
    for stale in PATHS.replay.glob("*.json"):
        stale.unlink()
    catalogue: list[dict] = []

    picks = (
        features.filter(F.col("week_index").isin(weeks))
        .groupBy("scenario_id", "scenario_type", "range_id", "account_id")
        .agg(F.max(LABEL_COLUMN).alias("has_onset"))
        .collect()
    )
    candidates: dict[str, list] = defaultdict(list)
    for row in picks:
        candidates[row["scenario_type"]].append(row)

    chosen = {
        scenario_type: choose_replays(
            rows, per_type, prefer_onset=scenario_type in ATTACK_SCENARIOS
        )
        for scenario_type, rows in sorted(candidates.items())
    }

    for scenario_type, rows in chosen.items():
        for row in rows:
            subset = (
                features.filter(
                    (features.scenario_id == row["scenario_id"])
                    & (features.range_id == row["range_id"])
                )
                .select(*REPLAY_COLUMNS)
                .orderBy("window_start_epoch", "traffic_type")
                .collect()
            )
            replay_id = f"{scenario_type}__{row['scenario_id']}"
            payload = {
                "replay_id": replay_id,
                "scenario_type": scenario_type,
                "scenario_id": row["scenario_id"],
                "range_id": row["range_id"],
                "has_onset": bool(row["has_onset"]),
                "windows": [r.asDict() for r in subset],
            }
            path: Path = PATHS.replay / f"{replay_id}.json"
            path.write_text(json.dumps(payload))
            catalogue.append(
                {
                    "replay_id": replay_id,
                    "scenario_type": scenario_type,
                    "range_id": row["range_id"],
                    "has_onset": bool(row["has_onset"]),
                    "window_count": len({r["window_start_epoch"] for r in subset}),
                }
            )

    (PATHS.replay / "catalogue.json").write_text(
        json.dumps(sorted(catalogue, key=lambda c: c["replay_id"]), indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DFSI Memory layer")
    parser.add_argument(
        "--days",
        type=int,
        default=28,
        help="days of history; the first week is baseline warm-up only",
    )
    parser.add_argument("--env", default=None, help="stage (default) or prod thresholds")
    parser.add_argument(
        "--replay-per-type",
        type=int,
        default=None,
        help="replays per scenario type (default: one per account, so every account is offered)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run(days=args.days, environment=args.env, replay_per_type=args.replay_per_type)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
