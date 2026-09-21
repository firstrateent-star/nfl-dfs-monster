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
import run_reality_loop_v721 as v721
import run_reality_loop_v722 as v722
import run_reality_loop_v723 as v723
from runtime_v724_composer import compose_v724_runtime, write_runtime_fingerprint_v724


def _record_v724_manifest(out) -> None:
    path = out / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "reality_engine_v724_active": True,
            "v724_base_architecture": "v7.2.3-left-tail-drive-resolution-calibration",
            "v724_left_tail_mass_unchanged": True,
            "v724_mode_gated_drive_consequence": True,
            "v724_normal_world_finishing_protected": True,
            "v724_hard_collapse_finishing_amplified": True,
            "v724_scoreboard_remains_event_derived": True,
            "market_inputs_used_for_football": False,
            "direct_fantasy_adjustment": False,
            "direct_score_adjustment": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
        }
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v724_runtime
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
    v721._record_v721_manifest(out)
    v722._record_v722_manifest(out)
    v723._record_v723_manifest(out)
    _record_v724_manifest(out)
    write_runtime_fingerprint_v724(out)


if __name__ == "__main__":
    main()
