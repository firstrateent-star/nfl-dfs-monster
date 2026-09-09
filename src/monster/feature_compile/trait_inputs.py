from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.skill_pools import compile_player_physical_inputs


def _value(row: dict, name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compile_player_trait_inputs(
    personnel: pl.DataFrame,
    *,
    game_date: date | None = None,
) -> dict[str, PlayerMechanismInputs]:
    """Compile all currently supported non-market skill-player trait evidence.

    Physical/biology inputs come from the existing nflverse/combine path. Madden attributes are
    scouting-style proxy evidence attached to the personnel snapshot and enter only the bounded
    football mechanism compiler. Missing Madden data is neutral rather than imputed as talent.
    """
    result = compile_player_physical_inputs(personnel, game_date=game_date)
    rows_by_id: dict[str, dict] = {}
    for row in personnel.to_dicts():
        player_id = str(row.get("gsis_id") or row.get("pfr_id") or "")
        if player_id:
            rows_by_id[player_id] = row

    for player_id, base in list(result.items()):
        row = rows_by_id.get(player_id, {})
        result[player_id] = replace(
            base,
            madden_speed=_value(row, "madden_speed"),
            madden_acceleration=_value(row, "madden_acceleration"),
            madden_route_running=_value(row, "madden_route_running"),
            madden_catching=_value(row, "madden_catching"),
        )
    return result


def trait_coverage(personnel: pl.DataFrame) -> dict[str, int]:
    """Machine-readable coverage for Full-Monster promotion manifests."""
    skill = personnel.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["QB", "RB", "WR", "TE"]))
    def count(column: str) -> int:
        return int(skill.select(pl.col(column).is_not_null().sum()).item()) if column in skill.columns else 0
    return {
        "skill_players": skill.height,
        "height": count("height"),
        "weight": count("weight"),
        "forty": count("forty"),
        "birth_date": count("birth_date"),
        "madden_speed": count("madden_speed"),
        "madden_acceleration": count("madden_acceleration"),
        "madden_route_running": count("madden_route_running"),
        "madden_catching": count("madden_catching"),
    }
