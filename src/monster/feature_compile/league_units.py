from __future__ import annotations

from dataclasses import asdict

import polars as pl

from monster.feature_compile.units import UnitPlayerInputs, compile_team_unit_effects


def _value(row: dict, key: str, default=None):
    value = row.get(key, default)
    return default if value is None else value


def unit_player_from_personnel_row(row: dict) -> UnitPlayerInputs:
    return UnitPlayerInputs(
        player_id=str(_value(row, "gsis_id", _value(row, "pfr_id", "unknown"))),
        position=str(_value(row, "position", _value(row, "depth_position", ""))),
        offense_snap_share=float(_value(row, "projected_offense_snap_share", 0.0)),
        defense_snap_share=float(_value(row, "projected_defense_snap_share", 0.0)),
        special_teams_snap_share=float(_value(row, "projected_special_teams_snap_share", 0.0)),
        snap_share_uncertainty=float(_value(row, "participation_uncertainty", 0.0)),
        active_probability=1.0,
        effectiveness_if_active=float(_value(row, "health_effectiveness_if_active", 1.0)),
        madden_pass_block=_value(row, "madden_pass_block"), madden_run_block=_value(row, "madden_run_block"),
        madden_pass_rush=_value(row, "madden_pass_rush"), madden_coverage=_value(row, "madden_coverage"),
        madden_tackle=_value(row, "madden_tackle"), madden_speed=_value(row, "madden_speed"),
        madden_acceleration=_value(row, "madden_acceleration"), madden_route_running=_value(row, "madden_route_running"),
        madden_catching=_value(row, "madden_catching"), madden_release=_value(row, "madden_release"),
        madden_carrying=_value(row, "madden_carrying"), madden_break_tackle=_value(row, "madden_break_tackle"),
        madden_awareness=_value(row, "madden_awareness"), madden_throw_power=_value(row, "madden_throw_power"),
        madden_throw_accuracy=_value(row, "madden_throw_accuracy"), madden_throw_under_pressure=_value(row, "madden_throw_under_pressure"),
        madden_ball_carrier_vision=_value(row, "madden_ball_carrier_vision"),
        madden_kick_power=_value(row, "madden_kick_power"), madden_kick_accuracy=_value(row, "madden_kick_accuracy"),
        madden_return=_value(row, "madden_return"), pass_block_signal=_value(row, "observed_pass_block_signal"),
        run_block_signal=_value(row, "observed_run_block_signal"), pass_rush_signal=_value(row, "observed_pass_rush_signal"),
        coverage_signal=_value(row, "observed_coverage_signal"), run_defense_signal=_value(row, "observed_run_defense_signal"),
        special_teams_signal=_value(row, "observed_special_teams_signal"),
    )


def compile_league_unit_player_map(snapshot: pl.DataFrame) -> dict[str, tuple[UnitPlayerInputs, ...]]:
    required = {"team_id", "projected_offense_snap_share", "projected_defense_snap_share", "projected_special_teams_snap_share", "participation_uncertainty"}
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"League unit compilation missing columns: {sorted(missing)}")
    return {
        str(team_id): tuple(unit_player_from_personnel_row(row) for row in snapshot.filter(pl.col("team_id") == team_id).to_dicts())
        for team_id in sorted(snapshot.get_column("team_id").unique().to_list())
    }


def _count_signal(team: pl.DataFrame, column: str) -> int:
    return int(team.select(pl.col(column).is_not_null().sum()).item()) if column in team.columns else 0


def compile_league_unit_effects(snapshot: pl.DataFrame) -> pl.DataFrame:
    player_map = compile_league_unit_player_map(snapshot)
    rows: list[dict] = []
    for team_id, players in player_map.items():
        team = snapshot.filter(pl.col("team_id") == team_id)
        effects, trace = compile_team_unit_effects(players)
        rows.append({
            "team_id": team_id, **asdict(effects), **asdict(trace), "roster_rows": team.height,
            "linemen_with_pass_block_signal": _count_signal(team, "observed_pass_block_signal"),
            "linemen_with_run_block_signal": _count_signal(team, "observed_run_block_signal"),
            "linemen_with_madden_pass_block": _count_signal(team, "madden_pass_block"),
            "linemen_with_madden_run_block": _count_signal(team, "madden_run_block"),
            "skill_players_with_madden_speed": _count_signal(team, "madden_speed"),
            "skill_players_with_madden_route": _count_signal(team, "madden_route_running"),
            "qbs_with_madden_accuracy": _count_signal(team, "madden_throw_accuracy"),
            "defenders_with_pass_rush_signal": _count_signal(team, "observed_pass_rush_signal"),
            "defenders_with_coverage_signal": _count_signal(team, "observed_coverage_signal"),
            "defenders_with_run_defense_signal": _count_signal(team, "observed_run_defense_signal"),
            "specialists_with_observed_signal": _count_signal(team, "observed_special_teams_signal"),
        })
    return pl.DataFrame(rows).sort("team_id")
