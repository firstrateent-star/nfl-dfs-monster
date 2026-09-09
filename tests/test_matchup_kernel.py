from __future__ import annotations

from monster.sim.matchup_kernel import (
    LEAGUE_PRESSURE_RATE,
    DefensiveIdentity,
    DefensiveUnit,
    resolve_pass_matchup,
    resolve_run_matchup,
)
from monster.sim.play_kernel import PlayerIdentity


def _defense(coverage: float = 1.0, rush: float = 1.0, run: float = 1.0) -> DefensiveUnit:
    front = (
        DefensiveIdentity("edge", "Edge", "EDGE", pass_rush=rush, run_defense=run),
    )
    secondary = (
        DefensiveIdentity("cb", "Corner", "CB", coverage=coverage, ball_hawk=coverage),
    )
    return DefensiveUnit(front=front, coverage=secondary)


def test_neutral_matchup_preserves_empirical_pressure_baseline() -> None:
    target = PlayerIdentity("wr", "WR", "WR")
    matchup = resolve_pass_matchup(
        target,
        _defense(),
        pass_protection=1.0,
        quarterback_efficiency=1.0,
    )
    assert abs(matchup.pressure_probability - LEAGUE_PRESSURE_RATE) < 1e-6


def test_better_coverage_reduces_completion_and_explosiveness() -> None:
    target = PlayerIdentity("wr", "WR", "WR", efficiency=1.1, explosive=1.1)
    weak = resolve_pass_matchup(target, _defense(coverage=0.8), pass_protection=1.0, quarterback_efficiency=1.0)
    elite = resolve_pass_matchup(target, _defense(coverage=1.25), pass_protection=1.0, quarterback_efficiency=1.0)
    assert elite.completion_probability < weak.completion_probability
    assert elite.yards_multiplier < weak.yards_multiplier
    assert elite.interception_probability > weak.interception_probability


def test_pass_rush_and_protection_interact_mechanically() -> None:
    target = PlayerIdentity("wr", "WR", "WR")
    strong_rush = resolve_pass_matchup(target, _defense(rush=1.3), pass_protection=0.85, quarterback_efficiency=1.0)
    protected = resolve_pass_matchup(target, _defense(rush=1.3), pass_protection=1.2, quarterback_efficiency=1.0)
    assert strong_rush.pressure_probability > protected.pressure_probability


def test_run_blocking_and_front_strength_move_run_matchup() -> None:
    back = PlayerIdentity("rb", "RB", "RB", efficiency=1.05)
    bad = resolve_run_matchup(back, _defense(run=1.25), run_blocking=0.85)
    good = resolve_run_matchup(back, _defense(run=0.9), run_blocking=1.15)
    assert bad.stuff_probability > good.stuff_probability
    assert bad.yards_multiplier < good.yards_multiplier