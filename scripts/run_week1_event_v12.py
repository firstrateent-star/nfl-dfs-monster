from __future__ import annotations

import importlib.util
from pathlib import Path

from monster.sim.pipeline_v12 import simulate_monster_game as simulate_event_monster_game


def _load_reality_runner():
    path = Path(__file__).with_name("run_week1_reality_v1.py")
    spec = importlib.util.spec_from_file_location("monster_week1_reality_v1_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Reality-v1 runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run Week 1 through the v1.2 event-first shadow engine.

    All Full-Reality evidence compilation remains identical to promoted v1. Only the causal
    simulation kernel changes: possessions -> finite plays -> field state -> scoring events ->
    inferred scoreboard. This is a shadow candidate and cannot promote itself.
    """
    reality = _load_reality_runner()
    original_loader = reality._load_baseline_runner

    def load_event_runner():
        runner = original_loader()
        runner.simulate_monster_game = simulate_event_monster_game
        return runner

    reality._load_baseline_runner = load_event_runner
    reality.main()


if __name__ == "__main__":
    main()
