from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from monster.feature_compile.mechanisms import PlayerMechanismInputs

_SKILL = {"QB", "RB", "WR", "TE"}
_TYPICAL_CAREER_VOLUME_PER_YEAR = {"QB": 500.0, "RB": 190.0, "WR": 105.0, "TE": 75.0}


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _age(value: object, game_date: date) -> float | None:
    if isinstance(value, datetime):
        born = value.date()
    elif isinstance(value, date):
        born = value
    elif isinstance(value, str) and value:
        try:
            born = date.fromisoformat(value[:10])
        except ValueError:
            return None
    else:
        return None
    return (game_date - born).days / 365.2425


def _career_volume_proxy(row: dict, position: str) -> float | None:
    """Conservative experience proxy used only for biology uncertainty, never expected scoring."""
    direct = _finite(row.get("career_workload"))
    if direct is not None:
        return direct
    years = _finite(row.get("years_of_experience"))
    if years is None:
        return None
    return max(years, 0.0) * _TYPICAL_CAREER_VOLUME_PER_YEAR[position]


def compile_player_reality_inputs(personnel: pl.DataFrame, *, game_date: date) -> dict[str, PlayerMechanismInputs]:
    """Compile current non-market human + Madden evidence into player mechanism inputs."""
    result: dict[str, PlayerMechanismInputs] = {}
    for row in personnel.to_dicts():
        position = str(row.get("position") or "").upper()
        if position not in _SKILL:
            continue
        player_id = str(row.get("gsis_id") or row.get("pfr_id") or "")
        if not player_id:
            continue
        result[player_id] = PlayerMechanismInputs(
            height_in=_finite(row.get("height")),
            weight_lbs=_finite(row.get("weight")),
            wingspan_in=_finite(row.get("wingspan")),
            forty_time=_finite(row.get("forty")),
            madden_speed=_finite(row.get("madden_speed")),
            madden_acceleration=_finite(row.get("madden_acceleration")),
            madden_route_running=_finite(row.get("madden_route_running")),
            madden_catching=_finite(row.get("madden_catching")),
            age_years=_age(row.get("birth_date"), game_date),
            career_workload=_career_volume_proxy(row, position),
            unit_continuity=_finite(row.get("unit_continuity")),
            active_probability=_finite(
                row.get("health_availability_probability", row.get("game_day_active_probability"))
            ),
            effectiveness_if_active=_finite(row.get("health_effectiveness_if_active")),
        )
    return result
