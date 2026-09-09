from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import polars as pl

from monster.feature_compile.mechanisms import TeamMechanismInputs, _weather_effect
from monster.feature_compile.reality_inputs import compile_player_reality_inputs

_ENVIRONMENT = Path("config/environment/week1_2026_2026-09-09.csv")


def _load_baseline_runner():
    path = Path(__file__).with_name("run_week1_full_monster.py")
    spec = importlib.util.spec_from_file_location("monster_week1_structural_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load structural runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_map() -> dict[str, dict]:
    if not _ENVIRONMENT.exists():
        return {}
    return {str(row["team_id"]): row for row in pl.read_csv(_ENVIRONMENT).to_dicts()}


def main() -> None:
    """Run the conserved structural kernel with Full-Reality player and environment inputs."""
    runner = _load_baseline_runner()
    environment = _environment_map()
    original_strengthened_state = runner._strengthened_state

    def full_inputs(personnel, *, game_date=None):
        if game_date is None:
            raise ValueError("Full-Reality runner requires an explicit game_date")
        return compile_player_reality_inputs(personnel, game_date=game_date)

    def strengthened_with_environment(base, unit_players, ol_row):
        state = original_strengthened_state(base, unit_players, ol_row)
        row = environment.get(str(state.team_id))
        if row is None:
            return state
        inputs = TeamMechanismInputs(
            wind_mph=row.get("wind_mph"),
            precipitation_probability=row.get("precipitation_probability"),
            temperature_f=row.get("temperature_f"),
            dome=bool(row.get("dome", False)),
        )
        return replace(state, weather_effect=_weather_effect(inputs))

    runner.compile_player_physical_inputs = full_inputs
    runner._strengthened_state = strengthened_with_environment
    runner.main()


if __name__ == "__main__":
    main()
