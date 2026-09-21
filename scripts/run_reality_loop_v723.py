from __future__ import annotations

import json
from pathlib import Path

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
from runtime_v723_composer import compose_v723_runtime, write_runtime_fingerprint_v723


def _record_v723_manifest(out: Path) -> None:
    path = out / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "reality_engine_v723_active": True,
            "v723_base_architecture": "v7.2.2-world-binary-availability",
            "v723_existing_role_uncertainty_reused": True,
            "v723_coherent_game_day_latents_active": True,
            "v723_world_team_mechanism_routing_active": True,
            "v723_world_defense_mechanism_routing_active": True,
            "v723_direct_score_adjustment": False,
            "v723_direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "v723_principle": (
                "Worlds may disagree about game-day execution and availability before "
                "play randomness begins. Existing role uncertainty remains authoritative; "
                "V7.2.3 does not create a second role sampler."
            ),
        }
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v723_runtime
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
    _record_v723_manifest(out)
    write_runtime_fingerprint_v723(out)


if __name__ == "__main__":
    main()
