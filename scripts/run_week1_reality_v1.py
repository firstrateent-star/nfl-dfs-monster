from __future__ import annotations

import importlib.util
from pathlib import Path

from monster.feature_compile.reality_inputs import compile_player_reality_inputs


def _load_baseline_runner():
    path = Path(__file__).with_name("run_week1_full_monster.py")
    spec = importlib.util.spec_from_file_location("monster_week1_structural_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load structural runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run the structural engine with the Full-Reality player input compiler.

    The structural runner remains the conserved simulation kernel. This wrapper replaces only its
    legacy physical-only player input compiler so current height/weight/forty/age AND Madden skill
    traits cross into the same causal player mechanisms before worlds are generated.
    """
    runner = _load_baseline_runner()

    def full_inputs(personnel, *, game_date=None):
        if game_date is None:
            raise ValueError("Full-Reality runner requires an explicit game_date")
        return compile_player_reality_inputs(personnel, game_date=game_date)

    runner.compile_player_physical_inputs = full_inputs
    runner.main()


if __name__ == "__main__":
    main()
