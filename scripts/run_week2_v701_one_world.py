from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import run_week1_v13_first_sim as first_sim
import run_week1_v13_integrated as integrated

WEEK2_MATCHUPS = (
    ("DET", "BUF"),
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
WEEK2_GAME_DATE = date(2026, 9, 20)


def _arg_path(flag: str) -> Path | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1])


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
            "games": len(WEEK2_MATCHUPS),
            "week2_game_date_reference": WEEK2_GAME_DATE.isoformat(),
            "week2_matchups": [f"{away}@{home}" for away, home in WEEK2_MATCHUPS],
            "week2_truth_used_for_priors": False,
            "week1_real_football_allowed_in_priors": True,
            "market_blind_football": True,
            "experiment": "v7.1-one-world-week2-2026",
            "view_semantics": (
                "Exactly one simulated universe per game; values are realized "
                "single-world box scores, not expectations."
            ),
        }
    )
    if kind == "first":
        payload["promotion_status"] = "EXPLORATORY_ONE_WORLD_NOT_CALIBRATION"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # Patch the legacy Week-1 runner's schedule boundary only. The v7.1 football
    # runtime itself is unchanged.
    first_sim.MATCHUPS = WEEK2_MATCHUPS
    first_sim.GAME_DATE = WEEK2_GAME_DATE
    integrated.MATCHUPS = WEEK2_MATCHUPS
    integrated.GAME_DATE = WEEK2_GAME_DATE

    import run_reality_loop_v701 as v701

    v701.main()

    _rewrite_manifest(_arg_path("--first-out"), kind="first")
    _rewrite_manifest(_arg_path("--box-out"), kind="box")


if __name__ == "__main__":
    main()
