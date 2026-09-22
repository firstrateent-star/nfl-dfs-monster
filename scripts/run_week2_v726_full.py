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
REFERENCE_DATE = date(2026, 9, 17)


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
            "week": 2,
            "games": len(WEEK2_MATCHUPS),
            "reference_date": REFERENCE_DATE.isoformat(),
            "input_cutoff_date": "2026-09-16",
            "matchups": [f"{away}@{home}" for away, home in WEEK2_MATCHUPS],
            "experiment": "v7.2.6-full-week2-preopener-2026",
            "week1_results_allowed_as_prior_evidence": True,
            "week2_truth_used_for_priors": False,
            "week2_current_snaps_used": False,
            "market_blind_football": True,
            "audit_control_release": "release/v726-verified-baseline",
        }
    )
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    first_sim.MATCHUPS = WEEK2_MATCHUPS
    first_sim.GAME_DATE = REFERENCE_DATE
    integrated.MATCHUPS = WEEK2_MATCHUPS
    integrated.GAME_DATE = REFERENCE_DATE

    import run_reality_loop_v726 as v726

    v726.main()
    _rewrite_manifest(_arg_path("--first-out"))
    _rewrite_manifest(_arg_path("--box-out"))


if __name__ == "__main__":
    main()
