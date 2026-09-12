from __future__ import annotations

from monster.sim.state_transition_eval import (
    ABSORB_CONVERTED,
    ABSORB_FAILED,
    coarse_state_key,
    distance_bucket,
    field_zone,
    replace_transition_family,
    solve_series_survival,
    total_variation,
)


def test_state_buckets_are_stable() -> None:
    assert distance_bucket(1.0) == "1"
    assert distance_bucket(3.0) == "2_3"
    assert distance_bucket(6.0) == "4_6"
    assert distance_bucket(10.0) == "7_10"
    assert distance_bucket(11.0) == "11_plus"
    assert field_zone(19.9) == "own_1_19"
    assert field_zone(80.0) == "red_zone"
    assert coarse_state_key(3, 7.0) == "d3:7_10"


def test_total_variation_matches_simple_distribution_gap() -> None:
    left = {"a": 0.75, "b": 0.25}
    right = {"a": 0.25, "b": 0.75}
    assert total_variation(left, right) == 0.5


def test_series_survival_solves_penalty_replay_loop() -> None:
    transitions = {
        "d1:7_10": {"d2:4_6": 0.60, ABSORB_CONVERTED: 0.20, "d1:7_10": 0.20},
        "d2:4_6": {ABSORB_CONVERTED: 0.50, ABSORB_FAILED: 0.50},
    }
    survival, values = solve_series_survival(transitions, {"d1:7_10": 1.0})
    assert abs(values["d2:4_6"] - 0.50) < 1e-9
    assert abs(survival - 0.625) < 1e-9


def test_replacing_one_family_changes_only_requested_states() -> None:
    baseline = {
        "d1:7_10": {ABSORB_FAILED: 1.0},
        "d2:4_6": {ABSORB_FAILED: 1.0},
    }
    replacement = {
        "d1:7_10": {ABSORB_CONVERTED: 1.0},
        "d2:4_6": {ABSORB_CONVERTED: 1.0},
    }
    hybrid = replace_transition_family(baseline, replacement, ["d2:4_6"])
    assert hybrid["d1:7_10"] == {ABSORB_FAILED: 1.0}
    assert hybrid["d2:4_6"] == {ABSORB_CONVERTED: 1.0}
