from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v636_v638 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Keep the Week 1 scoring target frozen; only the football runtime changes.
    base.main()

    out = _arg_path("--out")
    old_report = out / "v638_week1_paired_comparison.json"
    new_report = out / "v700_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v7.0-participation-first-paired-week1"
    payload["control_architecture"] = "frozen-v6.3.8"
    payload["challenger_architecture"] = "v7.0-participation-first-shadow"
    payload["single_mechanism_change"] = (
        "After the exact 11v11 participant world is known, target and designed-run "
        "assignment no longer use pre-sampled usage share as fallback authority. "
        "Historical concept evidence may tilt assignment among live participants; "
        "the v6 usage-based opportunity-skill tail is disabled. Pregame participation, "
        "coaching, matchup execution, scoring, environment and all market-blind inputs remain frozen."
    )
    payload["promotion_status"] = "SHADOW_NOT_PROMOTED"
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
