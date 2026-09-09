from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import polars as pl

from monster.feature_compile.environment import apply_environment_to_pool
from monster.feature_compile.mechanisms import TeamMechanismInputs, _weather_effect
from monster.feature_compile.reality_inputs import compile_player_reality_inputs

_ENVIRONMENT = Path("config/environment/week1_2026_2026-09-09.csv")
_ALLOWED_ABLATIONS = {"human", "madden", "units", "environment"}


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
    return {
        str(row["team_id"]): row
        for row in pl.read_csv(_ENVIRONMENT).to_dicts()
    }


def _disabled_families() -> set[str]:
    raw = os.environ.get("MONSTER_DISABLE_REALITY_FAMILIES", "").strip()
    disabled = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = disabled - _ALLOWED_ABLATIONS
    if unknown:
        raise ValueError(f"Unknown Full-Reality ablation families: {sorted(unknown)}")
    return disabled


def _ablate_player_inputs(inputs, disabled: set[str]):
    changes = {}
    if "human" in disabled:
        changes.update(
            height_in=None,
            weight_lbs=None,
            wingspan_in=None,
            forty_time=None,
            age_years=None,
            career_workload=None,
        )
    if "madden" in disabled:
        changes.update(
            madden_speed=None,
            madden_acceleration=None,
            madden_route_running=None,
            madden_catching=None,
        )
    return replace(inputs, **changes) if changes else inputs


def _environment_inputs(environment: dict[str, dict], team_id: str):
    row = environment.get(str(team_id))
    if row is None:
        return None
    return TeamMechanismInputs(
        wind_mph=row.get("wind_mph"),
        precipitation_probability=row.get("precipitation_probability"),
        temperature_f=row.get("temperature_f"),
        dome=bool(row.get("dome", False)),
    )


def main() -> None:
    """Run the conserved structural kernel with Full-Reality inputs.

    `MONSTER_DISABLE_REALITY_FAMILIES` is an evidence-only ablation seam. It may
    disable human/physical player traits, Madden player traits, compiled team units,
    and/or environment. Health and continuity remain active current-state controls.
    The default is the unchanged Full-Reality production candidate.
    """
    runner = _load_baseline_runner()
    environment = _environment_map()
    disabled = _disabled_families()
    original_strengthened_state = runner._strengthened_state
    original_pool_compile = runner.compile_current_skill_pools

    def full_inputs(personnel, *, game_date=None):
        if game_date is None:
            raise ValueError("Full-Reality runner requires an explicit game_date")
        compiled = compile_player_reality_inputs(personnel, game_date=game_date)
        return {
            player_id: _ablate_player_inputs(inputs, disabled)
            for player_id, inputs in compiled.items()
        }

    def full_pool_compile(*args, **kwargs):
        pools = original_pool_compile(*args, **kwargs)
        if "environment" in disabled:
            return pools
        return {
            team_id: (
                apply_environment_to_pool(pool, inputs)
                if (inputs := _environment_inputs(environment, team_id)) is not None
                else pool
            )
            for team_id, pool in pools.items()
        }

    def strengthened_with_environment(base, unit_players, ol_row):
        state = (
            base
            if "units" in disabled
            else original_strengthened_state(base, unit_players, ol_row)
        )
        if "environment" in disabled:
            return state
        inputs = _environment_inputs(environment, str(state.team_id))
        if inputs is None:
            return state
        return replace(state, weather_effect=_weather_effect(inputs))

    runner.compile_player_physical_inputs = full_inputs
    runner.compile_current_skill_pools = full_pool_compile
    runner._strengthened_state = strengthened_with_environment
    runner.main()


if __name__ == "__main__":
    main()
