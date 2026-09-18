from __future__ import annotations

import json
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634
import run_reality_loop_v635 as v635
from runtime_v636_composer import (
    compose_v636_runtime,
    write_runtime_fingerprint_v636,
)

from monster.sim import resolution_ecology


def _record_v636_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v636_shadow_active": True,
            "v636_rb_only_rush_reallocation_active": True,
            "v636_qb_rush_mass_preserved_from_native_plan": True,
            "v636_wr_te_gadget_rush_mass_preserved_from_native_plan": True,
            "v636_receiver_starter_status_is_not_rush_evidence": True,
            "v636_current_rb_starter_admission_active": True,
            "v636_target_sampler_frozen_to_v635": True,
            "v636_live_pass_matchup_authority": float(
                resolution_ecology._SNAP_MATCHUP_AUTHORITY
            ),
            "v636_live_run_matchup_authority": float(
                resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY
            ),
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "runtime_authority_map_v636": {
                "participation_truth": "current_role_guard_v635 current roster/depth truth",
                "conditional_target_share": "frozen v6.3.5 target sampler",
                "qb_rush_mass": "native v6.3 rushing plan, preserved exactly",
                "wr_te_gadget_rush_mass": "native v6.3 rushing plan, preserved exactly",
                "conditional_rb_rush_share": (
                    "native v6.3 RB mass + current RB depth/snap reallocation"
                ),
                "qb_execution": "frozen v6.3.5 rich-QB live field-read path",
                "qb_mobility": "frozen v6.3.5 pressure-response path",
                "wr_cb_matchup": "frozen v6.3.5 snap matchup path",
                "play_calling": "frozen v6.3.5 game_flow_policy + intent_ecology",
                "scoreboard": "game_loop_v13 event-derived only",
            },
            "v636_principle": (
                "Participation evidence and opportunity evidence are role-specific. "
                "RB depth/snap truth may move RB workload, but receiving starter status "
                "cannot manufacture designed carries for WR/TE players."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v636_runtime
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original

    out = v61._first_out_path()
    v63._TARGET_ROLE_TELEMETRY.write(out)
    v634._IDENTITY_TELEMETRY.write(out)
    v63._record_v63_manifest(out)
    v634._record_v634_manifest(out)
    v635._record_v635_manifest(out)
    _record_v636_manifest(out)
    write_runtime_fingerprint_v636(out)


if __name__ == "__main__":
    main()
