from __future__ import annotations

from dataclasses import asdict

import polars as pl

from monster.feature_compile.units import UnitPlayerInputs, compile_team_unit_effects


def _value(row: dict, key: str, default=None):
    value = row.get(key, default)
    return default if value is None else value


def _unit_player_from_row(row: dict) -> UnitPlayerInputs:
    """Adapt one conserved personnel row to the simulation unit compiler.

    `projected_*_snap_share` is already expected participation after roster-status
    availability and team-level conservation. Therefore active_probability is fixed at
    1.0 here to prevent double-counting availability. The original availability prior
    remains preserved in the personnel artifact for later role-state sampling.
    """
    return UnitPlayerInputs(
        player_id=str(_value(row, "gsis_id", _value(row, "pfr_id", "unknown"))),
        position=str(_value(row, "position", _value(row, "depth_position", ""))),
        offense_snap_share=float(_value(row, "projected_offense_snap_share", 0.0)),
        defense_snap_share=float(_value(row, "projected_defense_snap_share", 0.0)),
        special_teams_snap_share=float(_value(row, "projected_special_teams_snap_share", 0.0)),
        snap_share_uncertainty=float(_value(row, "participation_uncertainty", 0.0)),
        active_probability=1.0,
        effectiveness_if_active=1.0,
        pass_rush_signal=_value(row, "observed_pass_rush_signal"),
        coverage_signal=_value(row, "observed_coverage_signal"),
        run_defense_signal=_value(row, "observed_run_defense_signal"),
    )


def compile_league_unit_effects(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Compile one auditable six-mechanism unit-effect row for every NFL team.

    At this stage observed defensive evidence is allowed authority. Offensive blocking
    and specialist capability remain neutral until a defensible source is attached;
    missing evidence is not silently imputed as bad performance.
    """
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

    rows: list[dict] = []
    for team_id in sorted(snapshot.get_column("team_id").unique().to_list()):
        team = snapshot.filter(pl.col("team_id") == team_id)
        players = tuple(_unit_player_from_row(row) for row in team.to_dicts())
        effects, trace = compile_team_unit_effects(players)
        rows.append(
            {
                "team_id": team_id,
                **asdict(effects),
                **asdict(trace),
                "roster_rows": team.height,
                "defenders_with_pass_rush_signal": int(
                    team.select(pl.col("observed_pass_rush_signal").is_not_null().sum()).item()
                    if "observed_pass_rush_signal" in team.columns
                    else 0
                ),
                "defenders_with_coverage_signal": int(
                    team.select(pl.col("observed_coverage_signal").is_not_null().sum()).item()
                    if "observed_coverage_signal" in team.columns
                    else 0
                ),
                "defenders_with_run_defense_signal": int(
                    team.select(pl.col("observed_run_defense_signal").is_not_null().sum()).item()
                    if "observed_run_defense_signal" in team.columns
                    else 0
                ),
            }
        )
    return pl.DataFrame(rows).sort("team_id")
