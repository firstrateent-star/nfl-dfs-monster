from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import run_week1_v13_first_sim as first_sim
import run_week1_v13_integrated as integrated

WEEK1_MATCHUPS = (
    ("NE", "SEA"),
    ("SF", "LAR"),
    ("CHI", "CAR"),
    ("TB", "CIN"),
    ("NO", "DET"),
    ("BUF", "HOU"),
    ("BAL", "IND"),
    ("CLE", "JAC"),
    ("ATL", "PIT"),
    ("NYJ", "TEN"),
    ("ARI", "LAC"),
    ("MIA", "LV"),
    ("GB", "MIN"),
    ("WAS", "PHI"),
    ("DAL", "NYG"),
    ("DEN", "KC"),
)
WEEK1_REFERENCE_DATE = date(2026, 9, 9)


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
            "week": 1,
            "games": len(WEEK1_MATCHUPS),
            "week1_reference_date": WEEK1_REFERENCE_DATE.isoformat(),
            "week1_matchups": [
                f"{away}@{home}" for away, home in WEEK1_MATCHUPS
            ],
            "week1_truth_used_for_priors": False,
            "current_2026_regular_season_snaps_used": False,
            "market_blind_football": True,
            "experiment": "v7.1-one-world-week1-strict-2026",
            "view_semantics": (
                "Exactly one deterministic simulated universe per Week 1 game; "
                "values are realized single-world box scores, not expectations."
            ),
        }
    )
    if kind == "first":
        payload["promotion_status"] = "EXPLORATORY_ONE_WORLD_REALITY_AUDIT"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # Patch only the schedule boundary. The authoritative v7.1 runtime composition
    # remains unchanged.
    first_sim.MATCHUPS = WEEK1_MATCHUPS
    first_sim.GAME_DATE = WEEK1_REFERENCE_DATE
    integrated.MATCHUPS = WEEK1_MATCHUPS
    integrated.GAME_DATE = WEEK1_REFERENCE_DATE

    import run_reality_loop_v701 as v701

    v701.main()

    _rewrite_manifest(_arg_path("--first-out"), kind="first")
    _rewrite_manifest(_arg_path("--box-out"), kind="box")


if __name__ == "__main__":
    main()
