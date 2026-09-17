from __future__ import annotations

from dataclasses import replace

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v634_matchup_identity_authority import (
    apply_matchup_identity_authority_v634,
)
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit, resolve_pass_matchup
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.rich_identity import RichPlayerIdentity
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _capability(player_id: str, position: str, rating: float) -> UnitPlayerInputs:
    kwargs = {
        "player_id": player_id,
        "position": position,
        "madden_speed": rating,
        "madden_acceleration": rating,
        "madden_awareness": rating,
    }
    if position == "QB":
        kwargs.update(
            {
                "madden_throw_accuracy": rating,
                "madden_throw_power": rating,
                "madden_throw_under_pressure": rating,
                "madden_throw_on_run": rating,
                "madden_play_action": rating,
                "madden_break_sack": rating,
            }
        )
    elif position == "RB":
        kwargs.update(
            {
                "madden_carrying": rating,
                "madden_break_tackle": rating,
                "madden_ball_carrier_vision": rating,
                "madden_juke": rating,
                "madden_change_of_direction": rating,
            }
        )
    elif position in {"WR", "TE"}:
        kwargs.update(
            {
                "madden_route_running": rating,
                "madden_catching": rating,
                "madden_release": rating,
                "madden_catch_in_traffic": rating,
            }
        )
    return UnitPlayerInputs(**kwargs)


def _world() -> tuple[TeamIdentity, TeamPlayerPool, dict[str, PlayerMechanismInputs], TeamState]:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("qb", "QB", "QB", "A", qb_pass_share=1.0),
            PlayerState("rb", "RB", "RB", "A", rush_share=0.80),
            PlayerState("wr", "WR", "WR", "A", target_share=0.35),
        ),
    )
    base = TeamIdentity(
        team_id="A",
        quarterback=PlayerIdentity("qb", "QB", "QB", usage_weight=1.0),
        rushers=(PlayerIdentity("rb", "RB", "RB", usage_weight=0.80),),
        receivers=(PlayerIdentity("wr", "WR", "WR", usage_weight=0.35),),
        pass_efficiency=1.0,
        rush_efficiency=1.0,
        pass_protection=1.04,
        run_blocking=0.96,
        neutral_pass_rate=0.58,
    )
    reality = {
        pid: PlayerMechanismInputs(active_probability=1.0, effectiveness_if_active=1.0)
        for pid in ("qb", "rb", "wr")
    }
    state = TeamState(team_id="A", opponent_id="B")
    return base, pool, reality, state


def _apply(*, qb: float = 82.0, rb: float = 82.0, wr: float = 82.0):
    base, pool, reality, state = _world()
    return apply_matchup_identity_authority_v634(
        base,
        pool=pool,
        reality=reality,
        unit_players=(
            _capability("qb", "QB", qb),
            _capability("rb", "RB", rb),
            _capability("wr", "WR", wr),
        ),
        state=state,
    )


def test_qb_capability_monotonically_moves_team_pass_environment() -> None:
    values = []
    for rating in (68.0, 78.0, 88.0, 96.0):
        _, trace = _apply(qb=rating)
        values.append(trace.pass_efficiency_after)
    assert values == sorted(values)
    assert values[-1] > values[0] * 1.08


def test_runner_capability_monotonically_moves_team_run_environment() -> None:
    values = []
    for rating in (68.0, 78.0, 88.0, 96.0):
        _, trace = _apply(rb=rating)
        values.append(trace.rush_efficiency_after)
    assert values == sorted(values)
    assert values[-1] > values[0] * 1.04


def test_rich_channels_survive_v634_team_authority_boundary() -> None:
    enhanced, _ = _apply(qb=94.0, rb=94.0, wr=94.0)
    receiver = enhanced.receivers[0]
    runner = enhanced.rushers[0]
    assert isinstance(receiver, RichPlayerIdentity)
    assert isinstance(runner, RichPlayerIdentity)
    assert receiver.route_separation_skill > 0.0
    assert receiver.catchpoint_skill > 0.0
    assert runner.rush_creation_skill > 0.0
    assert runner.open_field_skill > 0.0


def test_unit_amplification_preserves_direction_around_neutral() -> None:
    enhanced, trace = _apply()
    assert enhanced.pass_protection > trace.pass_protection_before > 1.0
    assert enhanced.run_blocking < trace.run_blocking_before < 1.0


def _defense(coverage: float) -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            DefensiveIdentity(
                "edge", "EDGE", "EDGE", pass_rush=1.0, run_defense=1.0, tackling=1.0
            ),
        ),
        coverage=(
            DefensiveIdentity(
                "cb", "CB", "CB", coverage=coverage, ball_hawk=coverage, speed=coverage
            ),
        ),
    )


def _receiver(skill: float) -> RichPlayerIdentity:
    return RichPlayerIdentity(
        "wr",
        "WR",
        "WR",
        efficiency=1.0,
        explosive=1.0,
        route_separation_skill=skill,
        catchpoint_skill=skill,
        speed_skill=skill,
        open_field_skill=skill,
    )


def test_receiver_corner_counterfactual_has_material_matchup_identity_span() -> None:
    worst = resolve_pass_matchup(
        _receiver(-0.8),
        _defense(1.24),
        pass_protection=1.0,
        quarterback_efficiency=1.0,
        responsibility_key="v634-worst",
    )
    best = resolve_pass_matchup(
        _receiver(0.8),
        _defense(0.78),
        pass_protection=1.0,
        quarterback_efficiency=1.0,
        responsibility_key="v634-best",
    )
    assert best.completion_probability > worst.completion_probability + 0.12
    assert best.yards_multiplier > worst.yards_multiplier * 1.20
    assert best.local_separation_edge > worst.local_separation_edge


def test_qb_swap_changes_pass_prior_without_touching_play_calling_identity() -> None:
    low, low_trace = _apply(qb=68.0)
    high, high_trace = _apply(qb=96.0)
    assert high.pass_efficiency > low.pass_efficiency
    assert high.neutral_pass_rate == low.neutral_pass_rate
    assert high_trace.neutral_pass_rate == low_trace.neutral_pass_rate
