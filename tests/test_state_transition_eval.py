from __future__ import annotations

from monster.sim.state_transition_eval import (
    ABSORB_CONVERTED,
    ABSORB_FAILED,
    aggregate_joint_transition,
    coarse_state_key,
    distance_bucket,
    field_zone,
    joint_component_probabilities,
    replace_transition_family,
    solve_series_survival,
    total_variation,
    transplant_component_transition,
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


def test_joint_component_probabilities_sum_to_one() -> None:
    rows = [
        {"component": "run", "next_state": "d2:4_6"},
        {"component": "run", "next_state": ABSORB_CONVERTED},
        {"component": "pass", "next_state": "d2:7_10"},
        {"component": "pass", "next_state": ABSORB_FAILED},
    ]
    joint = joint_component_probabilities(rows)
    aggregate = aggregate_joint_transition(joint)
    assert abs(sum(aggregate.values()) - 1.0) < 1e-12
    assert joint["run"]["d2:4_6"] == 0.25
    assert aggregate[ABSORB_FAILED] == 0.25


def test_mass_and_shape_transplant_replaces_component_probability() -> None:
    baseline = {
        "run": {"d2:7_10": 0.40, ABSORB_CONVERTED: 0.10},
        "pass": {"d2:4_6": 0.30, ABSORB_FAILED: 0.20},
    }
    replacement = {
        "run": {"d2:4_6": 0.15, ABSORB_CONVERTED: 0.15},
        "pass": {"d2:4_6": 0.50, ABSORB_FAILED: 0.20},
    }
    hybrid = transplant_component_transition(
        baseline,
        replacement,
        "run",
        mode="mass_and_shape",
    )
    assert abs(sum(hybrid.values()) - 1.0) < 1e-12
    assert abs(hybrid[ABSORB_CONVERTED] - 0.15) < 1e-12
    assert abs(hybrid["d2:4_6"] - 0.57) < 1e-12
    assert abs(hybrid[ABSORB_FAILED] - 0.28) < 1e-12


def test_shape_only_transplant_preserves_component_mass() -> None:
    baseline = {
        "run": {"d2:7_10": 0.40, ABSORB_CONVERTED: 0.10},
        "pass": {"d2:4_6": 0.30, ABSORB_FAILED: 0.20},
    }
    replacement = {
        "run": {"d2:4_6": 0.15, ABSORB_CONVERTED: 0.15},
    }
    hybrid = transplant_component_transition(
        baseline,
        replacement,
        "run",
        mode="shape_only",
    )
    assert abs(sum(hybrid.values()) - 1.0) < 1e-12
    assert abs(hybrid[ABSORB_CONVERTED] - 0.25) < 1e-12
    assert abs(hybrid["d2:4_6"] - 0.55) < 1e-12
    assert abs(hybrid[ABSORB_FAILED] - 0.20) < 1e-12
