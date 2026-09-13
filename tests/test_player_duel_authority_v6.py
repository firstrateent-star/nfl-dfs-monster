from __future__ import annotations

from types import SimpleNamespace

import pytest

from monster.sim.matchup_kernel import (
    DefensiveIdentity,
    DefensiveUnit,
    resolve_pass_matchup,
    resolve_run_matchup,
)
from monster.sim.resolution_ecology import _shrink_relative, _shrink_run_relative
from monster.sim.rich_identity import RichPlayerIdentity
from monster.sim.snap_ecology import register_team_units


def _receiver() -> RichPlayerIdentity:
    return RichPlayerIdentity(
        "wr-v6",
        "Receiver",
        "WR",
        usage_weight=0.30,
        efficiency=1.06,
        explosive=1.08,
        speed_skill=0.55,
        route_separation_skill=0.60,
        catchpoint_skill=0.45,
        open_field_skill=0.50,
    )


def _runner() -> RichPlayerIdentity:
    return RichPlayerIdentity(
        "rb-v6",
        "Runner",
        "RB",
        usage_weight=0.45,
        efficiency=1.05,
        explosive=1.07,
        speed_skill=0.35,
        rush_creation_skill=0.55,
        runner_power_skill=0.40,
        open_field_skill=0.45,
    )


def _coverage_unit(coverage: float, ball_hawk: float) -> DefensiveUnit:
    cb = DefensiveIdentity(
        "cb-v6",
        "Corner",
        "CB",
        coverage=coverage,
        ball_hawk=ball_hawk,
        speed=coverage,
        snap_weight=1.0,
    )
    edge = DefensiveIdentity(
        "edge-neutral",
        "Edge",
        "EDGE",
        pass_rush=1.0,
        run_defense=1.0,
        tackling=1.0,
        snap_weight=1.0,
    )
    return DefensiveUnit(front=(edge,), coverage=(cb,))


def test_assigned_cb_skill_materially_changes_same_receiver_duel() -> None:
    target = _receiver()
    weak = resolve_pass_matchup(
        target,
        _coverage_unit(0.74, 0.78),
        pass_protection=1.0,
        quarterback_efficiency=1.05,
        responsibility_key="v6-cb-duel",
    )
    elite = resolve_pass_matchup(
        target,
        _coverage_unit(1.26, 1.24),
        pass_protection=1.0,
        quarterback_efficiency=1.05,
        responsibility_key="v6-cb-duel",
    )

    assert weak.primary_defender_id == elite.primary_defender_id == "cb-v6"
    assert weak.completion_probability > elite.completion_probability + 0.10
    assert weak.yards_multiplier > elite.yards_multiplier * 1.15
    assert elite.interception_probability > weak.interception_probability


def _register_ol(target_id: str, pass_block: float = 80.0, run_block: float = 80.0) -> None:
    rows = [
        SimpleNamespace(
            player_id=target_id,
            position="WR",
            offense_snap_share=0.90,
        )
    ]
    for idx, position in enumerate(("LT", "LG", "C", "RG", "RT")):
        rows.append(
            SimpleNamespace(
                player_id=f"v6-ol-{idx}",
                position=position,
                offense_snap_share=0.96,
                pass_block_signal=None,
                run_block_signal=None,
                madden_pass_block=pass_block,
                madden_run_block=run_block,
                madden_awareness=82.0,
                madden_stamina=88.0,
            )
        )
    register_team_units("V6", rows)


def test_edge_vs_ol_duel_materially_changes_pressure_timing() -> None:
    target = _receiver()
    _register_ol(target.player_id, pass_block=79.0)
    cb = DefensiveIdentity("cb-neutral", "CB", "CB", coverage=1.0, snap_weight=1.0)

    weak_edge = DefensiveIdentity(
        "edge-v6", "EDGE", "EDGE", pass_rush=0.74, run_defense=1.0, snap_weight=1.0
    )
    elite_edge = DefensiveIdentity(
        "edge-v6", "EDGE", "EDGE", pass_rush=1.26, run_defense=1.0, snap_weight=1.0
    )
    weak = resolve_pass_matchup(
        target,
        DefensiveUnit(front=(weak_edge,), coverage=(cb,)),
        pass_protection=1.0,
        quarterback_efficiency=1.0,
        responsibility_key="v6-edge-duel",
    )
    elite = resolve_pass_matchup(
        target,
        DefensiveUnit(front=(elite_edge,), coverage=(cb,)),
        pass_protection=1.0,
        quarterback_efficiency=1.0,
        responsibility_key="v6-edge-duel",
    )

    assert weak.primary_rusher_id == elite.primary_rusher_id == "edge-v6"
    assert elite.pressure_probability > weak.pressure_probability + 0.08
    assert elite.time_to_pressure < weak.time_to_pressure


def test_box_defender_skill_materially_changes_same_run_lane() -> None:
    rusher = _runner()
    dt_weak = DefensiveIdentity(
        "dt-v6", "DT", "DT", run_defense=0.74, tackling=0.90, snap_weight=1.0
    )
    dt_elite = DefensiveIdentity(
        "dt-v6", "DT", "DT", run_defense=1.26, tackling=1.10, snap_weight=1.0
    )
    safety = DefensiveIdentity(
        "s-v6", "S", "S", coverage=1.0, tackling=1.0, snap_weight=1.0
    )

    weak = resolve_run_matchup(
        rusher,
        DefensiveUnit(front=(dt_weak,), coverage=(safety,)),
        run_blocking=1.0,
        responsibility_key="v6-run-duel",
        run_geometry="interior",
    )
    elite = resolve_run_matchup(
        rusher,
        DefensiveUnit(front=(dt_elite,), coverage=(safety,)),
        run_blocking=1.0,
        responsibility_key="v6-run-duel",
        run_geometry="interior",
    )

    assert weak.primary_defender_id == elite.primary_defender_id == "dt-v6"
    assert elite.stuff_probability > weak.stuff_probability + 0.04
    assert weak.yards_multiplier > elite.yards_multiplier * 1.12


def test_snap_matchup_authority_is_strong_but_neutral_centered() -> None:
    assert _shrink_relative(1.0, low=0.72, high=1.30) == pytest.approx(1.0)
    assert _shrink_run_relative(1.0, low=0.72, high=1.38) == pytest.approx(1.0)
    assert _shrink_relative(1.30, low=0.72, high=1.30) > 1.20
    assert _shrink_relative(0.72, low=0.72, high=1.30) < 0.80
    assert _shrink_run_relative(1.38, low=0.72, high=1.38) > 1.28
    assert _shrink_run_relative(0.72, low=0.72, high=1.38) < 0.80
