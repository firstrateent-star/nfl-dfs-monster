from __future__ import annotations

import json

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634
import run_reality_loop_v635 as v635
import run_reality_loop_v636 as v636
import run_reality_loop_v638 as v638
import run_reality_loop_v700 as v700
import run_reality_loop_v701 as v701
import run_reality_loop_v720 as v720
from runtime_v721_composer import (
    compose_v721_runtime,
    write_runtime_fingerprint_v721,
)


def _record_v721_manifest(out) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_engine_v721_shadow_active": True,
            "v721_base_architecture": "v7.2.0-shadow-reality-allocation",
            "v721_two_minute_clock_cadence_active": True,
            "v721_timeout_state_active": True,
            "v721_timeout_banks_per_half": 3,
            "v721_four_minute_clock_drain_active": True,
            "v721_late_multiscore_defensive_posture_active": True,
            "v721_blowout_skill_preservation_active": True,
            "v721_terminal_qb_preservation_active": True,
            "v721_game_script_telemetry": "game_script_v721.csv",
            "v721_preservation_telemetry": "preservation_v721.csv",
            "v721_pass_run_brain_changed": False,
            "v721_week2_truth_used_for_priors": False,
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "v721_principle": (
                "Clock, timeout, defensive shell and personnel-preservation effects "
                "must emerge from score/time football state. Final scores and fantasy "
                "outcomes have no authority over these mechanisms."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v721_runtime
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original

    out = v61._first_out_path()
    v63._TARGET_ROLE_TELEMETRY.write(out)
    v634._IDENTITY_TELEMETRY.write(out)
    v63._record_v63_manifest(out)
    v634._record_v634_manifest(out)
    v635._record_v635_manifest(out)
    v636._record_v636_manifest(out)
    v638._record_v638_manifest(out)
    v700._record_v700_manifest(out)
    v701._record_v701_manifest(out)
    v720._record_v720_manifest(out)
    _record_v721_manifest(out)
    write_runtime_fingerprint_v721(out)


if __name__ == "__main__":
    main()
