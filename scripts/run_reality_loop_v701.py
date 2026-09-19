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
from runtime_v701_composer import (
    compose_v701_runtime,
    write_runtime_fingerprint_v701,
)


def _record_v701_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_engine_v701_shadow_active": True,
            "v701_base_architecture": "v7.0.0-participation-first-shadow",
            "v701_qb_rush_family_separation_active": True,
            "v701_qb_total_rush_share_reserves_designed_mass": False,
            "v701_non_qb_rush_plan_renormalized_after_qb_removal": True,
            "v701_qb_kept_concept_eligible_with_no_share_authority": True,
            "v701_qb_non_sneak_run_requires_geometry_evidence": True,
            "v701_scramble_generated_only_from_dropback_response": True,
            "v701_target_world_changed": False,
            "v701_new_random_draws": False,
            "v701_week1_truth_used_for_priors": False,
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "DEPENDENT_SHADOW_NOT_LAUNCHED",
            "v701_principle": (
                "Aggregate quarterback rushing history cannot reserve designed-run "
                "workload because it mixes designed runs, scrambles and kneels. "
                "Designed QB opportunities must arise only from designed-run concept "
                "evidence; scrambles remain downstream consequences of dropbacks."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v701_runtime
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
    _record_v701_manifest(out)
    write_runtime_fingerprint_v701(out)


if __name__ == "__main__":
    main()
