from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_full_game_counterfactual_v635 as v635_audit
from runtime_v701_composer import (
    build_week1_runtime_inputs_v701,
    write_runtime_fingerprint_v701,
)


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    v635_audit.build_week1_runtime_inputs_v635 = build_week1_runtime_inputs_v701
    v635_audit.write_runtime_fingerprint_v635 = write_runtime_fingerprint_v701
    v635_audit.main()

    out = _arg_path("--out")
    old_worlds = out / "v635_full_game_qb_counterfactual_worlds.csv"
    new_worlds = out / "v701_full_game_qb_counterfactual_worlds.csv"
    if old_worlds.exists():
        old_worlds.replace(new_worlds)

    old_report = out / "v635_full_game_qb_counterfactual.json"
    new_report = out / "v701_full_game_qb_counterfactual.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v7.0.1-qb-rush-family-hierarchical-qb-counterfactual"
    payload["audit_logic"] = (
        "frozen hierarchical QB audit with authoritative v7.1 shadow runtime"
    )
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
