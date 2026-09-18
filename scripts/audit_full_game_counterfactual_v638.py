from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_full_game_counterfactual_v635 as v635_audit
from runtime_v638_composer import (
    build_week1_runtime_inputs_v638,
    write_runtime_fingerprint_v638,
)


def _arg_path(flag: str) -> Path:
    index = sys.argv.index(flag)
    return Path(sys.argv[index + 1])


def main() -> None:
    # Reuse the exact proven hierarchical-QB audit implementation, but inject the
    # authoritative v6.3.8 runtime so the diagnostic and production simulator execute
    # the same machine.
    v635_audit.build_week1_runtime_inputs_v635 = build_week1_runtime_inputs_v638
    v635_audit.write_runtime_fingerprint_v635 = write_runtime_fingerprint_v638
    v635_audit.main()

    out = _arg_path("--out")
    old_worlds = out / "v635_full_game_qb_counterfactual_worlds.csv"
    new_worlds = out / "v638_full_game_qb_counterfactual_worlds.csv"
    if old_worlds.exists():
        old_worlds.replace(new_worlds)

    old_report = out / "v635_full_game_qb_counterfactual.json"
    new_report = out / "v638_full_game_qb_counterfactual.json"
    payload = json.loads(old_report.read_text(encoding="utf-8"))
    payload["experiment"] = "v6.3.8-hierarchical-full-game-qb-counterfactual"
    payload["audit_logic"] = "exact v6.3.5 hierarchical audit, v6.3.8 runtime injected"
    new_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    old_report.unlink()


if __name__ == "__main__":
    main()
