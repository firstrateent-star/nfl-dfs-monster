from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v636_v638 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Reuse the exact Week 1 grading/comparison metric set. v6 closes without
    # moving the target or adding a new outcome-tuned objective.
    base.main()

    out = _arg_path("--out")
    old_report = out / "v638_week1_paired_comparison.json"
    new_report = out / "v639_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.9-final-paired-week1-reality"
    payload["control_architecture"] = "frozen-v6.3.8"
    payload["challenger_architecture"] = "v6.3.9-pretruth-gadget-entry-calibration"
    payload["single_mechanism_change"] = (
        "WR/TE designed-rush entry probability is replaced by a market-blind "
        "2022-2025 out-of-sample recurrence prior keyed to raw prior-season carry "
        "evidence and current rotation exposure; conditional carry workload, QB "
        "rush mass, RB hierarchy, targets, coaching, matchup identity, player "
        "execution and score construction remain frozen"
    )
    payload["v6_closeout_candidate"] = True
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
