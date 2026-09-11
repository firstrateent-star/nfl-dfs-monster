from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from monster.feature_compile.environment import apply_environment
from monster.feature_compile.mechanisms import TeamMechanismInputs
from monster.snapshot.model import TeamState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class V13EnvironmentTrace:
    home_team: str
    venue: str | None
    dome: bool
    surface: str | None
    temperature_f: float | None
    wind_mph: float | None
    precipitation_probability: float | None
    weather_effect: float
    source_note: str | None


def _float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def environment_inputs_for_home(
    environment: pl.DataFrame,
    *,
    home_team: str,
) -> tuple[TeamMechanismInputs, dict[str, object]]:
    """Resolve one game environment from the venue/home-team snapshot.

    The current Week 1 file is keyed by home team because venue/weather belong to the game,
    not separately to each offense. Both teams receive the same physical environment.
    Unknown future fields may be retained in the source snapshot without gaining authority
    until a mechanism explicitly consumes them.
    """

    if "team_id" not in environment.columns:
        raise ValueError("Environment snapshot requires team_id")
    rows = environment.filter(pl.col("team_id") == home_team).to_dicts()
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one environment row for home team {home_team}, found {len(rows)}"
        )
    row = rows[0]
    inputs = TeamMechanismInputs(
        wind_mph=_float(row.get("wind_mph")),
        precipitation_probability=_float(row.get("precipitation_probability")),
        temperature_f=_float(row.get("temperature_f")),
        dome=_bool(row.get("dome")),
    )
    return inputs, row


def apply_v13_game_environment(
    *,
    away_state: TeamState,
    home_state: TeamState,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    environment: pl.DataFrame,
    home_team: str,
) -> tuple[TeamState, TeamState, TeamPlayerPool, TeamPlayerPool, V13EnvironmentTrace]:
    """Apply one shared game environment at its existing football jurisdictions."""

    inputs, row = environment_inputs_for_home(environment, home_team=home_team)
    away_state, away_pool = apply_environment(away_state, away_pool, inputs)
    home_state, home_pool = apply_environment(home_state, home_pool, inputs)
    trace = V13EnvironmentTrace(
        home_team=home_team,
        venue=None if row.get("venue") is None else str(row.get("venue")),
        dome=inputs.dome,
        surface=None if row.get("surface") is None else str(row.get("surface")),
        temperature_f=inputs.temperature_f,
        wind_mph=inputs.wind_mph,
        precipitation_probability=inputs.precipitation_probability,
        weather_effect=away_state.weather_effect,
        source_note=None if row.get("source_note") is None else str(row.get("source_note")),
    )
    return away_state, home_state, away_pool, home_pool, trace
