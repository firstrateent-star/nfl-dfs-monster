from __future__ import annotations

import importlib.util
from pathlib import Path

from monster.feature_compile.environment import apply_environment_to_pool
from monster.sim.pipeline_v11 import simulate_monster_game as simulate_monster_game_v11


def _load(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fantasy = _load("run_week1_fantasy_complete.py")
    reality = _load("run_week1_reality_v1.py")

    # Reuse the certified Football Reality v1 compilation and environment mechanisms
    # inside the fantasy-complete runner. v1.1 then couples player TD outcomes to the
    # receptions/carries that actually occurred in each world. DFS remains downstream.
    environment = reality._environment_map()
    disabled: set[str] = set()
    original_state = fantasy._state
    original_pool_compile = fantasy.compile_current_skill_pools

    def full_inputs(personnel, *, game_date=None):
        if game_date is None:
            game_date = fantasy.GAME_DATE
        compiled = reality.compile_player_reality_inputs(personnel, game_date=game_date)
        return {
            pid: reality._ablate_player_inputs(inputs, disabled)
            for pid, inputs in compiled.items()
        }

    def environment_inputs(team_id: str):
        row = environment.get(str(team_id))
        if row is None:
            return None
        return reality.TeamMechanismInputs(
            wind_mph=row.get("wind_mph"),
            precipitation_probability=row.get("precipitation_probability"),
            temperature_f=row.get("temperature_f"),
            dome=bool(row.get("dome", False)),
        )

    def full_pool_compile(*args, **kwargs):
        pools = original_pool_compile(*args, **kwargs)
        return {
            team_id: (
                apply_environment_to_pool(pool, inputs)
                if (inputs := environment_inputs(team_id)) is not None
                else pool
            )
            for team_id, pool in pools.items()
        }

    def full_state(base, unit_players, ol_row):
        state = original_state(base, unit_players, ol_row)
        inputs = environment_inputs(str(state.team_id))
        if inputs is None:
            return state
        return reality.replace(state, weather_effect=reality._weather_effect(inputs))

    fantasy.compile_player_physical_inputs = full_inputs
    fantasy.compile_current_skill_pools = full_pool_compile
    fantasy.simulate_monster_game = simulate_monster_game_v11
    fantasy._state = full_state
    fantasy.main()


if __name__ == "__main__":
    main()
