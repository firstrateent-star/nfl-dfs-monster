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
REFERENCE_DATE = date(2026, 9, 9)


def _arg_path(flag: str) -> Path | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1]) if index + 1 < len(sys.argv) else None


def _rewrite_manifest(path: Path | None) -> None:
    if path is None or not (path / "manifest.json").exists():
        return
    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "season": 2026,
            "week": 1,
            "games": len(WEEK1_MATCHUPS),
            "reference_date": REFERENCE_DATE.isoformat(),
            "matchups": [f"{away}@{home}" for away, home in WEEK1_MATCHUPS],
            "experiment": "v7.2.6-full-week1-strict-2026",
            "week1_truth_used_for_priors": False,
            "current_2026_regular_season_snaps_used": False,
            "market_blind_football": True,
            "audit_control_release": "release/v726-verified-baseline",
        }
    )
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    first_sim.MATCHUPS = WEEK1_MATCHUPS
    first_sim.GAME_DATE = REFERENCE_DATE
    integrated.MATCHUPS = WEEK1_MATCHUPS
    integrated.GAME_DATE = REFERENCE_DATE

    import run_reality_loop_v726 as v726

    v726.main()
    _rewrite_manifest(_arg_path("--first-out"))
    _rewrite_manifest(_arg_path("--box-out"))


if __name__ == "__main__":
    main()
