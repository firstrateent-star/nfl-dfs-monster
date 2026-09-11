from dataclasses import dataclass

from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity
from monster.sim.snap_ecology import (
    register_team_units,
    resolve_pass_snap,
    resolve_run_snap,
)


@dataclass
class Unit:
    player_id: str
    position: str
    offense_snap_share: float = 1.0
    madden_pass_block: float | None = None
    madden_run_block: float | None = None
    madden_awareness: float | None = None
    madden_stamina: float | None = None
    madden_route_running: float | None = None
    madden_catching: float | None = None
    pass_block_signal: float | None = None
    run_block_signal: float | None = None


def _line(pass_block: float, run_block: float) -> list[Unit]:
    return [
        Unit(f"ol{i}", pos, madden_pass_block=pass_block, madden_run_block=run_block, madden_awareness=82, madden_stamina=84)
        for i, pos in enumerate(("LT", "LG", "C", "RG", "RT"))
    ]


def _defense(*, rush: float = 1.0, coverage: float = 1.0, run_defense: float = 1.0) -> DefensiveUnit:
    return DefensiveUnit(
        front=tuple(
            DefensiveIdentity(
                f"dl{i}",
                f"DL{i}",
                "EDGE" if i == 0 else "DT",
                pass_rush=rush,
                run_defense=run_defense,
                tackling=run_defense,
                snap_weight=0.85,
            )
            for i in range(4)
        ),
        coverage=(
            DefensiveIdentity("cb1", "CB1", "CB", coverage=coverage, ball_hawk=coverage, tackling=1.0, snap_weight=0.95),
            DefensiveIdentity("cb2", "CB2", "CB", coverage=coverage * 0.98, ball_hawk=coverage, tackling=1.0, snap_weight=0.90),
            DefensiveIdentity("fs", "FS", "FS", coverage=coverage * 1.02, ball_hawk=coverage, tackling=1.03, snap_weight=0.92),
            DefensiveIdentity("ss", "SS", "SS", coverage=coverage, ball_hawk=coverage, tackling=1.05, snap_weight=0.88),
        ),
    )


def test_five_ol_individually_change_pressure_timing() -> None:
    target = PlayerIdentity("wr", "WR", "WR", efficiency=1.0, explosive=1.0)
    weak_units = _line(68, 74) + [target]
    register_team_units("W", weak_units)
    weak = resolve_pass_snap(target=target, defense=_defense(rush=1.15), pass_protection=1.0, quarterback_efficiency=1.0)

    strong_units = _line(94, 88) + [target]
    register_team_units("S", strong_units)
    strong = resolve_pass_snap(target=target, defense=_defense(rush=1.15), pass_protection=1.0, quarterback_efficiency=1.0)

    assert strong.time_to_pressure > weak.time_to_pressure
    assert strong.pressure_probability < weak.pressure_probability
    assert strong.pocket_integrity > weak.pocket_integrity


def test_rb_or_te_help_improves_worst_protection_lane() -> None:
    target = PlayerIdentity("wrhelp", "WR", "WR")
    register_team_units("N", _line(76, 80) + [target])
    no_help = resolve_pass_snap(target=target, defense=_defense(rush=1.20), pass_protection=1.0, quarterback_efficiency=1.0)

    helper = Unit("te", "TE", offense_snap_share=0.8, madden_pass_block=92, madden_run_block=86, madden_route_running=72, madden_catching=76)
    register_team_units("H", _line(76, 80) + [helper, target])
    with_help = resolve_pass_snap(target=target, defense=_defense(rush=1.20), pass_protection=1.0, quarterback_efficiency=1.0)

    assert with_help.protection_help > no_help.protection_help
    assert with_help.pressure_probability <= no_help.pressure_probability


def test_safety_help_and_zone_overlap_reduce_read_quality() -> None:
    target = PlayerIdentity("wrzone", "WR", "WR", efficiency=1.08, explosive=1.12)
    register_team_units("Z", _line(84, 84) + [target])
    soft = resolve_pass_snap(target=target, defense=_defense(coverage=0.78), pass_protection=1.0, quarterback_efficiency=1.05)
    tight = resolve_pass_snap(target=target, defense=_defense(coverage=1.20), pass_protection=1.0, quarterback_efficiency=1.05)

    assert tight.safety_help >= soft.safety_help
    assert tight.zone_overlap >= soft.zone_overlap
    assert tight.coverage_strength > soft.coverage_strength
    assert tight.qb_read_quality < soft.qb_read_quality


def test_run_gap_uses_line_front_and_second_level() -> None:
    runner = PlayerIdentity("rb", "RB", "RB", efficiency=1.05, explosive=1.08)
    register_team_units("R1", _line(92, 94) + [runner])
    good_line = resolve_run_snap(rusher=runner, defense=_defense(run_defense=0.95), run_blocking=1.0)

    register_team_units("R2", _line(70, 68) + [runner])
    bad_line = resolve_run_snap(rusher=runner, defense=_defense(run_defense=1.15), run_blocking=1.0)

    assert good_line.lane_blocking > bad_line.lane_blocking
    assert good_line.front_fit < bad_line.front_fit
    assert good_line.stuff_probability < bad_line.stuff_probability
    assert good_line.yards_multiplier > bad_line.yards_multiplier
