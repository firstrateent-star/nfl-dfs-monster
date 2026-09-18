from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v634_v635 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Reuse the exact same grading/comparison metric set to avoid moving the goalposts.
    base.main()

    out = _arg_path("--out")
    old_report = out / "v635_week1_paired_comparison.json"
    new_report = out / "v636_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.6-paired-week1-reality"
    payload["control_architecture"] = "frozen-v6.3.5"
    payload["challenger_architecture"] = "v6.3.6-rb-only-role-reallocation"
    payload["single_mechanism_change"] = (
        "current depth/snap reweights RB workload only; "
        "native QB and WR/TE rush mass preserved"
    )
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
