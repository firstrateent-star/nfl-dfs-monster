from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import run_week1_v13_first_sim as first_sim
import run_week1_v13_integrated as integrated

MATCHUPS = (("NYG", "LAR"),)
GAME_DATE = date(2026, 9, 21)
INFORMATION_CUTOFF = date(2026, 9, 21)


def _arg_path(flag: str) -> Path | None:
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    return Path(sys.argv[i + 1]) if i + 1 < len(sys.argv) else None


def _arg_int(flag: str, default: int) -> int:
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    return int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else default


def _rewrite_manifest(path: Path | None) -> None:
    if path is None:
        return
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text())
    payload.update(
        {
            "season": 2026,
            "week": 2,
            "games": 1,
            "worlds_per_game": _arg_int("--worlds", 2000),
            "information_cutoff": INFORMATION_CUTOFF.isoformat(),
            "matchups": ["NYG@LAR"],
            "single_game_showdown": True,
            "week2_results_allowed": False,
            "market_blind_football": True,
            "experiment": "v7.2.6-series-survival-ecology-mnf",
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    first_sim.MATCHUPS = MATCHUPS
    first_sim.GAME_DATE = GAME_DATE
    integrated.MATCHUPS = MATCHUPS
    integrated.GAME_DATE = GAME_DATE

    import run_reality_loop_v726 as v726

    v726.main()
    _rewrite_manifest(_arg_path("--first-out"))
    _rewrite_manifest(_arg_path("--box-out"))


if __name__ == "__main__":
    main()
