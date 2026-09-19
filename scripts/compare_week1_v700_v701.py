from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v638_v700 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    base.main()

    out = _arg_path("--out")
    old_report = out / "v700_week1_paired_comparison.json"
    new_report = out / "v701_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v7.0.1-qb-rush-family-paired-week1"
    payload["control_architecture"] = "v7.0.0-participation-first-control"
    payload["challenger_architecture"] = "v7.0.1-qb-rush-family-shadow"
    payload["single_mechanism_change"] = (
        "Aggregate QB rush_share no longer reserves designed-run workload in the "
        "world role simplex. Non-QB workload is renormalized; the QB remains concept-"
        "eligible and can receive designed carries only through clean non-dropback "
        "run-geometry evidence or the explicit sneak concept. Scrambles remain solely "
        "downstream QB responses on dropbacks. Target roles, participation-first "
        "assignment, coaching, matchup execution, scoring and environment remain frozen."
    )
    payload["promotion_status"] = "DEPENDENT_SHADOW_NOT_PROMOTED"
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
