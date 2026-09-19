from __future__ import annotations

import json
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634
import run_reality_loop_v635 as v635
import run_reality_loop_v636 as v636
import run_reality_loop_v638 as v638
from runtime_v700_composer import (
    compose_v700_runtime,
    write_runtime_fingerprint_v700,
)


def _record_v700_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_engine_v700_shadow_active": True,
            "v700_base_architecture": "frozen-v6.3.8",
            "v700_participation_first_opportunity_active": True,
            "v700_pregame_target_role_used_for_participation_only": True,
            "v700_pregame_rush_role_used_for_participation_only": True,
            "v700_post_snap_assignment_uses_usage_weight": False,
            "v700_usage_based_opportunity_skill_tail_active": False,
            "v700_current_11v11_participants_authoritative": True,
            "v700_historical_concept_evidence_can_tilt_live_assignment": True,
            "v700_qb_non_sneak_designed_run_requires_geometry_evidence": True,
            "v700_week1_truth_used_for_priors": False,
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "V7_SHADOW_NOT_PROMOTED",
            "v700_principle": (
                "Historical role evidence may help a player reach the snap, but after the "
                "participants are known, target and designed-run ownership emerge from live "
                "participation, concept compatibility, matchup, skill and QB read quality "
                "rather than a preallocated final share."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v700_runtime
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
    _record_v700_manifest(out)
    write_runtime_fingerprint_v700(out)


if __name__ == "__main__":
    main()
