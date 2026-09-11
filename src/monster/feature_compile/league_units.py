from __future__ import annotations

from dataclasses import asdict

import polars as pl

from monster.feature_compile.units import UnitPlayerInputs, compile_team_unit_effects
from monster.sim.snap_ecology import register_team_units


def _value(row: dict, key: str, default=None):
    value = row.get(key, default)
    return default if value is None else value


def unit_player_from_personnel_row(
    row: dict,
    *,
    conditional_world: bool = False,
) -> UnitPlayerInputs:
    """Compile one player's unit evidence for expected-state or sampled-world use.

    Expected-state projected snap shares already contain availability and are conserved to the
    unit, so their active_probability must remain 1.0. For a sampled game world, use the
    pre-availability conditional snap shares together with the raw game-day active probability;
    the availability sampler will decide whether the player exists and then reconserve the unit.
    """
    if conditional_world:
        offense_share = float(_value(row, "conditional_offense_snap_share", 0.0))
        defense_share = float(_value(row, "conditional_defense_snap_share", 0.0))
        special_share = float(_value(row, "conditional_special_teams_snap_share", 0.0))
        active_probability = float(_value(row, "game_day_active_probability", 1.0))
    else:
        offense_share = float(_value(row, "projected_offense_snap_share", 0.0))
        defense_share = float(_value(row, "projected_defense_snap_share", 0.0))
        special_share = float(_value(row, "projected_special_teams_snap_share", 0.0))
        active_probability = 1.0

    return UnitPlayerInputs(
        player_id=str(_value(row, "gsis_id", _value(row, "pfr_id", "unknown"))),
        position=str(_value(row, "position", _value(row, "depth_position", ""))),
        offense_snap_share=offense_share,
        defense_snap_share=defense_share,
        special_teams_snap_share=special_share,
        snap_share_uncertainty=float(_value(row, "participation_uncertainty", 0.0)),
        active_probability=active_probability,
        effectiveness_if_active=float(_value(row, "health_effectiveness_if_active", 1.0)),
        madden_pass_block=_value(row, "madden_pass_block"),
        madden_run_block=_value(row, "madden_run_block"),
        madden_pass_rush=_value(row, "madden_pass_rush"),
        madden_coverage=_value(row, "madden_coverage"),
        madden_tackle=_value(row, "madden_tackle"),
        madden_speed=_value(row, "madden_speed"),
        madden_acceleration=_value(row, "madden_acceleration"),
        madden_route_running=_value(row, "madden_route_running"),
        madden_catching=_value(row, "madden_catching"),
        madden_catch_in_traffic=_value(row, "madden_catch_in_traffic"),
        madden_spectacular_catch=_value(row, "madden_spectacular_catch"),
        madden_release=_value(row, "madden_release"),
        madden_carrying=_value(row, "madden_carrying"),
        madden_break_tackle=_value(row, "madden_break_tackle"),
        madden_strength=_value(row, "madden_strength"),
        madden_agility=_value(row, "madden_agility"),
        madden_change_of_direction=_value(row, "madden_change_of_direction"),
        madden_awareness=_value(row, "madden_awareness"),
        madden_throw_power=_value(row, "madden_throw_power"),
        madden_throw_accuracy=_value(row, "madden_throw_accuracy"),
        madden_throw_under_pressure=_value(row, "madden_throw_under_pressure"),
        madden_throw_on_run=_value(row, "madden_throw_on_run"),
        madden_play_action=_value(row, "madden_play_action"),
        madden_break_sack=_value(row, "madden_break_sack"),
        madden_ball_carrier_vision=_value(row, "madden_ball_carrier_vision"),
        madden_juke=_value(row, "madden_juke"),
        madden_spin=_value(row, "madden_spin"),
        madden_stiff_arm=_value(row, "madden_stiff_arm"),
        madden_trucking=_value(row, "madden_trucking"),
        madden_jump=_value(row, "madden_jump"),
        madden_injury=_value(row, "madden_injury"),
        madden_stamina=_value(row, "madden_stamina"),
        madden_kick_power=_value(row, "madden_kick_power"),
        madden_kick_accuracy=_value(row, "madden_kick_accuracy"),
        madden_return=_value(row, "madden_return"),
        pass_block_signal=_value(row, "observed_pass_block_signal"),
        run_block_signal=_value(row, "observed_run_block_signal"),
        pass_rush_signal=_value(row, "observed_pass_rush_signal"),
        coverage_signal=_value(row, "observed_coverage_signal"),
        run_defense_signal=_value(row, "observed_run_defense_signal"),
        special_teams_signal=_value(row, "observed_special_teams_signal"),
    )


def _register_player_map(player_map: dict[str, tuple[UnitPlayerInputs, ...]]) -> None:
    for team_id, players in player_map.items():
        register_team_units(team_id, players)


def compile_league_unit_player_map(
    snapshot: pl.DataFrame,
) -> dict[str, tuple[UnitPlayerInputs, ...]]:
    """Compile expected-state unit inputs and freeze them for snap-level matchup physics."""
    required = {
        "team_id",
        "projected_offense_snap_share",
        "projected_defense_snap_share",
        "projected_special_teams_snap_share",
        "participation_uncertainty",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"League unit compilation missing columns: {sorted(missing)}")
    player_map = {
        str(team_id): tuple(
            unit_player_from_personnel_row(row)
            for row in snapshot.filter(pl.col("team_id") == team_id).to_dicts()
        )
        for team_id in sorted(snapshot.get_column("team_id").unique().to_list())
    }
    _register_player_map(player_map)
    return player_map


def compile_league_conditional_unit_player_map(
    snapshot: pl.DataFrame,
) -> dict[str, tuple[UnitPlayerInputs, ...]]:
    """Compile pre-availability unit inputs for one-world personnel sampling."""
    required = {
        "team_id",
        "conditional_offense_snap_share",
        "conditional_defense_snap_share",
        "conditional_special_teams_snap_share",
        "game_day_active_probability",
        "participation_uncertainty",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(
            f"Conditional unit compilation missing columns: {sorted(missing)}"
        )
    player_map = {
        str(team_id): tuple(
            unit_player_from_personnel_row(row, conditional_world=True)
            for row in snapshot.filter(pl.col("team_id") == team_id).to_dicts()
        )
        for team_id in sorted(snapshot.get_column("team_id").unique().to_list())
    }
    _register_player_map(player_map)
    return player_map


def _count_signal(team: pl.DataFrame, column: str) -> int:
    return (
        int(team.select(pl.col(column).is_not_null().sum()).item())
        if column in team.columns
        else 0
    )


def compile_league_unit_effects(snapshot: pl.DataFrame) -> pl.DataFrame:
    player_map = compile_league_unit_player_map(snapshot)
    rows: list[dict] = []
    for team_id, players in player_map.items():
        team = snapshot.filter(pl.col("team_id") == team_id)
        effects, trace = compile_team_unit_effects(players)
        rows.append(
            {
                "team_id": team_id,
                **asdict(effects),
                **asdict(trace),
                "roster_rows": team.height,
                "linemen_with_pass_block_signal": _count_signal(
                    team, "observed_pass_block_signal"
                ),
                "linemen_with_run_block_signal": _count_signal(
                    team, "observed_run_block_signal"
                ),
                "linemen_with_madden_pass_block": _count_signal(
                    team, "madden_pass_block"
                ),
                "linemen_with_madden_run_block": _count_signal(team, "madden_run_block"),
                "skill_players_with_madden_speed": _count_signal(team, "madden_speed"),
                "skill_players_with_madden_route": _count_signal(
                    team, "madden_route_running"
                ),
                "qbs_with_madden_accuracy": _count_signal(team, "madden_throw_accuracy"),
                "defenders_with_pass_rush_signal": _count_signal(
                    team, "observed_pass_rush_signal"
                ),
                "defenders_with_coverage_signal": _count_signal(
                    team, "observed_coverage_signal"
                ),
                "defenders_with_run_defense_signal": _count_signal(
                    team, "observed_run_defense_signal"
                ),
                "specialists_with_observed_signal": _count_signal(
                    team, "observed_special_teams_signal"
                ),
            }
        )
    return pl.DataFrame(rows).sort("team_id")