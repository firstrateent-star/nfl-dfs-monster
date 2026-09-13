from __future__ import annotations

import json
import sys
from pathlib import Path

import run_reality_loop_v2_smoke as reality
import run_week1_v13_integrated as integrated


def _pop_arg(flag: str, default: str) -> str:
    """Consume one wrapper-only CLI flag before delegating to the integrated runner."""
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    sys.argv.pop(idx)
    if idx >= len(sys.argv):
        return default
    return sys.argv.pop(idx)


def main() -> None:
    away = _pop_arg("--away", "DAL").upper()
    home = _pop_arg("--home", "NYG").upper()

    # The integrated runner imports MATCHUPS by value, so patch that production binding
    # before Reality Loop v6.1 config delegates into it. This preserves the exact same
    # football engine while concentrating all Monte Carlo worlds on one matchup.
    integrated.MATCHUPS = ((away, home),)

    reality.main()

    out = reality._first_out_path()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "single_game_mode": True,
                "single_game_away": away,
                "single_game_home": home,
                "single_game_matchup": f"{away}@{home}",
                "single_game_engine": "Reality Loop v6.1 player-duel authority",
                "single_game_concentrated_worlds": True,
                "football_is_source_of_truth": True,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
