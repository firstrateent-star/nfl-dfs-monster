from __future__ import annotations

from monster.sim.matchup_kernel import (
    DefensiveIdentity,
    DefensiveUnit,
    resolve_pass_matchup,
    resolve_run_matchup,
)
from monster.sim.play_kernel import PlayerIdentity


def _defense(*, coverage: float = 1.0, rush: float = 1.0, run_defense: float = 1.0) -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            DefensiveIdentity(
                player_id="edge",
                name="EDGE",
                position="EDGE",
                pass_rush=rush,
                run_defense=run_defense,
                tackling=run_defense,
                snap_weight=0.85,
            ),
        ),
        coverage=(
            DefensiveIdentity(
                player_id="cb",
                name="CB",
                position="CB",
                coverage=coverage,
                ball_hawk=coverage,
                tackling=coverage,
                snap_weight=0.92,
            ),
        ),
    )


def test_receiver_quality_changes_same_defender_matchup() -> None:
    weak = PlayerIdentity("wr1", "Weak", "WR", efficiency=0.82, explosive=0.92)
    strong = PlayerIdentity("wr2", "Strong", "WR", efficiency=1.18, explosive=1.12)
    defense = _defense(coverage=1.08)
    weak_matchup = resolve_pass_matchup(
        weak, defense, pass_protection=1.0, quarterback_efficiency=1.0
    )
    strong_matchup = resolve_pass_matchup(
        strong, defense, pass_protection=1.0, quarterback_efficiency=1.0
    )
    assert strong_matchup.local_separation_edge > weak_matchup.local_separation_edge
    assert strong_matchup.completion_probability > weak_matchup.completion_probability
    assert strong_matchup.yards_multiplier > weak_matchup.yards_multiplier


def test_corner_quality_changes_same_receiver_matchup() -> None:
    receiver = PlayerIdentity("wr", "WR", "WR", efficiency=1.05, explosive=1.05)
    soft = resolve_pass_matchup(
        receiver, _defense(coverage=0.82), pass_protection=1.0, quarterback_efficiency=1.0
    )
    elite = resolve_pass_matchup(
        receiver, _defense(coverage=1.20), pass_protection=1.0, quarterback_efficiency=1.0
    )
    assert elite.local_separation_edge < soft.local_separation_edge
    assert elite.completion_probability < soft.completion_probability
    assert elite.interception_probability > soft.interception_probability


def test_individual_edge_rush_changes_pressure_inside_same_protection() -> None:
    receiver = PlayerIdentity("wr", "WR", "WR")
    weak_rush = resolve_pass_matchup(
        receiver, _defense(rush=0.78), pass_protection=1.0, quarterback_efficiency=1.0
    )
    elite_rush = resolve_pass_matchup(
        receiver, _defense(rush=1.24), pass_protection=1.0, quarterback_efficiency=1.0
    )
    assert elite_rush.local_rush_strength > weak_rush.local_rush_strength
    assert elite_rush.pressure_probability > weak_rush.pressure_probability


def test_rusher_quality_changes_box_duel_and_yards() -> None:
    weak = PlayerIdentity("rb1", "Weak", "RB", efficiency=0.84)
    strong = PlayerIdentity("rb2", "Strong", "RB", efficiency=1.18)
    defense = _defense(run_defense=1.08)
    weak_matchup = resolve_run_matchup(weak, defense, run_blocking=1.0)
    strong_matchup = resolve_run_matchup(strong, defense, run_blocking=1.0)
    assert strong_matchup.runner_edge > weak_matchup.runner_edge
    assert strong_matchup.stuff_probability < weak_matchup.stuff_probability
    assert strong_matchup.yards_multiplier > weak_matchup.yards_multiplier
