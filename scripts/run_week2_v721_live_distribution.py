from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import run_week1_v13_first_sim as first_sim
import run_week1_v13_integrated as integrated

LIVE_WEEK2_MATCHUPS = (
    ("CAR", "ATL"),
    ("NO", "BAL"),
    ("MIN", "CHI"),
    ("CIN", "HOU"),
    ("PIT", "NE"),
    ("GB", "NYJ"),
    ("CLE", "TB"),
    ("PHI", "TEN"),
    ("JAC", "DEN"),
    ("LV", "LAC"),
    ("SEA", "ARI"),
    ("WAS", "DAL"),
    ("MIA", "SF"),
    ("IND", "KC"),
    ("NYG", "LAR"),
)
GAME_DATE = date(2026, 9, 20)
INFORMATION_CUTOFF = date(2026, 9, 19)


def _arg_path(flag: str) -> Path | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1])


def _arg_int(flag: str, default: int) -> int:
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        return default
    return int(sys.argv[index + 1])


def _rewrite_manifest(path: Path | None, *, kind: str) -> None:
    if path is None:
        return
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "season": 2026,
            "week": 2,
            "games": len(LIVE_WEEK2_MATCHUPS),
            "worlds_per_game": _arg_int("--worlds", 120),
            "information_cutoff": INFORMATION_CUTOFF.isoformat(),
            "matchups": [
                f"{away}@{home}" for away, home in LIVE_WEEK2_MATCHUPS
            ],
            "det_buf_excluded_already_played": True,
            "week2_game_results_used_for_priors": False,
            "week1_real_football_allowed_in_priors": True,
            "week2_current_snaps_allowed": False,
            "live_roster_depth_injury_information_allowed": True,
            "market_blind_football": True,
            "experiment": "v7.2.1-live-week2-2026-sunday-distribution",
        }
    )
    if kind == "first":
        payload["promotion_status"] = "LIVE_PREGAME_FORECAST_NOT_CALIBRATION"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    first_sim.MATCHUPS = LIVE_WEEK2_MATCHUPS
    first_sim.GAME_DATE = GAME_DATE
    integrated.MATCHUPS = LIVE_WEEK2_MATCHUPS
    integrated.GAME_DATE = GAME_DATE

    import run_reality_loop_v721 as v721

    v721.main()

    _rewrite_manifest(_arg_path("--first-out"), kind="first")
    _rewrite_manifest(_arg_path("--box-out"), kind="box")


if __name__ == "__main__":
    main()
