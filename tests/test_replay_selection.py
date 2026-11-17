"""Which scenarios become dashboard replays: every account must be offered.

Regression: picking two per family in scenario-id order spread the replays over
ranges but not accounts, so both landed on the first account's two ranges and
the dashboard's account dropdown showed a single account.
"""

from __future__ import annotations

from dfsi.memory.pipeline import choose_replays

#: Three accounts; account 1 owns ranges 10 and 11, and has the lowest ids.
ROWS = [
    {"scenario_id": f"s{n:02d}", "account_id": account, "range_id": rng, "has_onset": onset}
    for n, (account, rng, onset) in enumerate(
        [(1, 10, True), (1, 11, True), (1, 10, False), (2, 20, False), (2, 21, True), (3, 30, True)]
    )
]


def test_by_default_every_account_gets_a_replay():
    picked = choose_replays(ROWS, None, prefer_onset=False)
    assert sorted(r["account_id"] for r in picked) == [1, 2, 3]


def test_a_cap_still_spreads_across_accounts_before_ranges():
    picked = choose_replays(ROWS, 2, prefer_onset=False)
    assert {r["account_id"] for r in picked} == {1, 2}  # not both of account 1's ranges


def test_attack_families_prefer_scenarios_whose_bell_rings():
    picked = choose_replays(ROWS, None, prefer_onset=True)
    assert all(r["has_onset"] for r in picked)
    assert next(r for r in picked if r["account_id"] == 2)["range_id"] == 21


def test_after_accounts_it_spreads_across_ranges_then_fills():
    picked = choose_replays(ROWS, 5, prefer_onset=False)
    assert len({r["range_id"] for r in picked}) == 5
    assert len(choose_replays(ROWS, 6, prefer_onset=False)) == 6
    assert len(choose_replays(ROWS, 99, prefer_onset=False)) == len(ROWS)


def test_selection_is_deterministic():
    assert choose_replays(ROWS, None, False) == choose_replays(list(reversed(ROWS)), None, False)
