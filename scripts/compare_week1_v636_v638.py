from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v635_v636 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Reuse the exact same Week 1 metric set so the grading target does not move.
    base.main()

    out = _arg_path("--out")
    old_report = out / "v636_week1_paired_comparison.json"
    new_report = out / "v638_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.8-paired-week1-reality"
    payload["control_architecture"] = "frozen-v6.3.6"
    payload["challenger_architecture"] = "v6.3.8-gadget-rush-decomposition"
    payload["single_mechanism_change"] = (
        "WR/TE gadget rushing is decomposed into current rotation exposure, "
        "rush_role_probability entry, and 2022-2025 empirical conditional carry "
        "counts; QB mass, RB internal hierarchy, targets, coaching, matchup identity "
        "and player-skill execution remain frozen"
    )
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
