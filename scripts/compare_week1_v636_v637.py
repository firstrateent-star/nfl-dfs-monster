from __future__ import annotations

import json
import sys
from pathlib import Path

import compare_week1_v635_v636 as base


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    base.main()

    out = _arg_path("--out")
    old_report = out / "v636_week1_paired_comparison.json"
    new_report = out / "v637_week1_paired_comparison.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.7-paired-week1-reality"
    payload["control_architecture"] = "frozen-v6.3.6"
    payload["challenger_architecture"] = "v6.3.7-wr-te-gadget-entry"
    payload["single_mechanism_change"] = (
        "WR/TE players must win designed-rush admission through their existing "
        "rush_role_probability; rejected gadget mass returns proportionally to the "
        "frozen v6.3.6 RB hierarchy; QB mass, RB ordering, targets, coaching and "
        "player-skill identity remain frozen"
    )
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
