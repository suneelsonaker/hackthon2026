"""Policy store — the simulated policy database, and replays re-run under it.

`raw.json` lists, per account, range and traffic type, the configured policy
(`max_threshold_pps` / `max_threshold_mbps`). The API loads it once into this
in-memory store. When a human approves a suggested threshold change, the store
is updated — a new version, with history — and every replay served afterwards
is re-derived under the *current* policy:

* the wall drops whatever exceeds the new line (`blocked_pps` / `blocked_mbps`);
* the policy-dependent features the forecaster reads (`blocked_share` and its
  deltas, `policy_utilisation`) are recomputed;
* Shared State's view is recomputed: over-threshold windows, the declaration
  after the start duration, and the onset.

So replaying the same attack after an approval shows the new policy in effect:
more of the attack dropped at the wall, the bell ringing on the new line, and
the forecaster scoring traffic against the new policy.

Scope and safety: this is a simulated store inside DFSI. Nothing here reaches
Behemoth, Shared State or a customer's real configuration. The store is
in-memory — restarting the API, or `reset()`, restores `raw.json`.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import HORIZON_WINDOWS, SharedStateThresholds, thresholds_for

log = logging.getLogger("dfsi.policy_store")

_WINDOWS_PER_MINUTE = 6

#: Bytes per packet, used to convert a pps line to Mbps when raw.json does not
#: say. Matches the generator's assumptions.
_DEFAULT_PACKET_BYTES = {"udp": 1100, "tcp_syn": 64, "http": 700}


@dataclass(frozen=True)
class TypePolicy:
    pps: float
    mbps: float
    mean_packet_bytes: int


class PolicyStore:
    """Thread-safe, versioned, in-memory policy table seeded from raw.json."""

    def __init__(self, seed: dict | None = None, source: str | None = None) -> None:
        self._lock = threading.Lock()
        self._seed = copy.deepcopy(seed or {"accounts": []})
        self.source = source
        self._reset_locked()

    # ------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: Path) -> PolicyStore:
        if not path.exists():
            log.warning("no policy db at %s; replays keep their recorded policy", path)
            return cls(None, source=None)
        return cls(json.loads(path.read_text()), source=str(path))

    def _reset_locked(self) -> None:
        self._policies: dict[int, dict[str, TypePolicy]] = {}
        self._accounts: dict[int, int] = {}
        self._range_types: dict[int, str] = {}
        for account in self._seed.get("accounts", []):
            for rng in account.get("ranges", []):
                range_id = int(rng["range_id"])
                self._accounts[range_id] = int(account["account_id"])
                self._range_types[range_id] = str(rng.get("range_type", ""))
                self._policies[range_id] = {
                    traffic_type: TypePolicy(
                        pps=float(spec["max_threshold_pps"]),
                        mbps=float(spec["max_threshold_mbps"]),
                        mean_packet_bytes=int(
                            spec.get(
                                "mean_packet_bytes", _DEFAULT_PACKET_BYTES.get(traffic_type, 1000)
                            )
                        ),
                    )
                    for traffic_type, spec in rng.get("traffic_types", {}).items()
                }
        self.version = 1
        self.history: list[dict] = []

    def reset(self) -> int:
        """Restore the seed (raw.json) and start a new version."""
        with self._lock:
            previous = self.version
            self._reset_locked()
            self.version = previous + 1
            self.history.append(
                {"version": self.version, "at_epoch": int(time.time()), "action": "reset"}
            )
            return self.version

    # ------------------------------------------------------------ reading
    def range_policy(self, range_id: int) -> dict[str, TypePolicy] | None:
        with self._lock:
            policy = self._policies.get(int(range_id))
            return dict(policy) if policy is not None else None

    def range_version(self, range_id: int) -> int:
        """Version of the newest change that touched this range (1 = seed)."""
        with self._lock:
            touched = [
                h["version"]
                for h in self.history
                if h.get("range_id") == int(range_id) or h.get("action") == "reset"
            ]
            return max(touched) if touched else 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "source": self.source,
                "version": self.version,
                "ranges": [
                    {
                        "range_id": range_id,
                        "account_id": self._accounts.get(range_id),
                        "range_type": self._range_types.get(range_id),
                        "traffic_types": {
                            t: {"max_threshold_pps": p.pps, "max_threshold_mbps": round(p.mbps, 3)}
                            for t, p in sorted(types.items())
                        },
                    }
                    for range_id, types in sorted(self._policies.items())
                ],
                "history": list(self.history),
            }

    # ------------------------------------------------------------ writing
    def apply_change(
        self, range_id: int, new_pps: dict[str, float], provenance: dict | None = None
    ) -> int:
        """Set new per-type policy lines on a range. Returns the new version.

        Mbps lines move with the pps line, at the type's packet size.
        """
        range_id = int(range_id)
        with self._lock:
            policy = self._policies.get(range_id)
            if policy is None:
                raise KeyError(f"range {range_id} is not in the policy store")
            # Validate everything first: a bad line must not leave the earlier
            # ones applied with no version bump and no history entry.
            for traffic_type, pps in new_pps.items():
                if traffic_type not in policy:
                    raise KeyError(f"range {range_id} has no {traffic_type} policy")
                if not pps > 0:
                    raise ValueError(f"policy line must be positive, got {pps}")
            changes = []
            for traffic_type, pps in new_pps.items():
                old = policy[traffic_type]
                if float(pps) == old.pps:
                    continue  # not a change: no new version, no history entry
                mbps = pps * old.mean_packet_bytes * 8 / 1_000_000
                policy[traffic_type] = TypePolicy(float(pps), float(mbps), old.mean_packet_bytes)
                changes.append(
                    {
                        "traffic_type": traffic_type,
                        "from_pps": old.pps,
                        "to_pps": float(pps),
                        "from_mbps": round(old.mbps, 3),
                        "to_mbps": round(mbps, 3),
                    }
                )
            if not changes:
                return self.version
            self.version += 1
            self.history.append(
                {
                    "version": self.version,
                    "at_epoch": int(time.time()),
                    "action": "change",
                    "range_id": range_id,
                    "changes": changes,
                    **(provenance or {}),
                }
            )
            return self.version


# --------------------------------------------------------------------------- #
# Re-deriving a replay under a policy
# --------------------------------------------------------------------------- #
def _policy_matches(rows: Iterable[dict], policy: dict[str, TypePolicy]) -> bool:
    for row in rows:
        spec = policy.get(str(row.get("traffic_type")))
        if spec is None:
            continue
        if abs(float(row.get("policy_pps", 0.0)) - spec.pps) > 1e-6:
            return False
        if abs(float(row.get("policy_mbps", 0.0)) - spec.mbps) > 1e-6:
            return False
    return True


def apply_policy(
    windows: Sequence[dict],
    policy: dict[str, TypePolicy] | None,
    thresholds: SharedStateThresholds | None = None,
    force: bool = False,
) -> list[dict]:
    """Replay rows as they would have been recorded under `policy`.

    Traffic itself (`total_*`, growth, baseline deviation, source mix) does not
    depend on the policy and is kept. Everything the wall and Shared State
    derive from the policy is recomputed, with the same rules the feature
    pipeline uses. If the policy is what the rows were recorded under, the rows
    are returned unchanged (unless `force`).
    """
    if policy is None or (not force and _policy_matches(windows, policy)):
        return list(windows)
    t = thresholds or thresholds_for()

    rows = [dict(row) for row in windows]
    for row in rows:
        spec = policy.get(str(row.get("traffic_type")))
        if spec is not None:
            row["policy_pps"] = spec.pps
            row["policy_mbps"] = spec.mbps
        total_pps = float(row.get("total_pps", 0.0))
        total_mbps = float(row.get("total_mbps", 0.0))
        policy_pps = float(row["policy_pps"])
        policy_mbps = float(row["policy_mbps"])
        blocked_pps = max(0.0, total_pps - policy_pps)
        blocked_mbps = max(0.0, total_mbps - policy_mbps)
        row["blocked_pps"] = blocked_pps
        row["blocked_mbps"] = blocked_mbps
        row["blocked_share"] = blocked_pps / total_pps if total_pps else 0.0
        row["policy_utilisation"] = total_pps / policy_pps if policy_pps else 0.0
        row["policy_utilisation_mbps"] = total_mbps / policy_mbps if policy_mbps else 0.0
        row["over_threshold"] = int(
            blocked_pps >= policy_pps * t.start_threshold_ratio
            and blocked_pps >= t.start_floor_kpps * 1_000.0
            and blocked_mbps >= t.start_floor_mbps
        )

    # Per-type trailing deltas (same as features: a missing lag counts as 0).
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("traffic_type")), []).append(row)
    for series in by_type.values():
        series.sort(key=lambda r: int(r["window_start_epoch"]))
        shares = [r["blocked_share"] for r in series]
        for i, row in enumerate(series):
            prev_10 = shares[i - 1] if i >= 1 else 0.0
            prev_60 = shares[i - _WINDOWS_PER_MINUTE] if i >= _WINDOWS_PER_MINUTE else 0.0
            row["blocked_share_delta_10s"] = row["blocked_share"] - prev_10
            row["blocked_share_delta_60s"] = row["blocked_share"] - prev_60

    # Shared State, per instant: over threshold -> declared after the start
    # duration -> onset at the first declared window of a run.
    by_epoch: dict[int, list[dict]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["window_start_epoch"]), []).append(row)
    epochs = sorted(by_epoch)
    over = [max(int(r["over_threshold"]) for r in by_epoch[e]) for e in epochs]
    need = t.start_duration_windows
    declared = [
        int(i + 1 >= need and sum(over[i - need + 1 : i + 1]) >= need) for i in range(len(epochs))
    ]
    onset = [int(d == 1 and (i == 0 or declared[i - 1] == 0)) for i, d in enumerate(declared)]
    for i, epoch in enumerate(epochs):
        for row in by_epoch[epoch]:
            row["range_over_threshold"] = over[i]
            row["sustained_windows"] = sum(over[max(0, i - need + 1) : i + 1])
            row["declared"] = declared[i]
            row["is_onset"] = onset[i]

    # Per-type label: an onset this type carries within the horizon.
    index = {e: i for i, e in enumerate(epochs)}
    for series in by_type.values():
        carries = [
            onset[index[int(r["window_start_epoch"])]] * int(r["over_threshold"]) for r in series
        ]
        for i, row in enumerate(series):
            row["label_onset_120s"] = int(any(carries[i + 1 : i + 1 + HORIZON_WINDOWS]))
    return rows
