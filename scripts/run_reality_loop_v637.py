from __future__ import annotations

import json
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634
import run_reality_loop_v635 as v635
import run_reality_loop_v636 as v636
from runtime_v637_composer import (
    compose_v637_runtime,
    write_runtime_fingerprint_v637,
)

from monster.sim import resolution_ecology


def _record_v637_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v637_shadow_active": True,
            "v637_gadget_entry_gate_active": True,
            "v637_wr_te_entry_uses_existing_rush_role_probability": True,
            "v637_rejected_gadget_mass_returns_to_existing_rb_hierarchy": True,
            "v637_qb_rush_mass_frozen_to_v636": True,
            "v637_rb_internal_hierarchy_frozen_to_v636": True,
            "v637_target_sampler_frozen_to_v635": True,
            "v637_live_pass_matchup_authority": float(
                resolution_ecology._SNAP_MATCHUP_AUTHORITY
            ),
            "v637_live_run_matchup_authority": float(
                resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY
            ),
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "runtime_authority_map_v637": {
                "participation_truth": "current roster/depth truth inherited from v6.3.6",
                "conditional_target_share": "frozen v6.3.5 target sampler",
                "qb_rush_mass": "frozen v6.3.6 rushing plan",
                "wr_te_rush_entry": (
                    "existing rush_role_probability sampled as designed-carry-tree admission"
                ),
                "wr_te_rush_magnitude_if_admitted": "native v6.3.6 gadget share",
                "conditional_rb_rush_share": (
                    "frozen v6.3.6 RB hierarchy; absorbs rejected gadget residual proportionally"
                ),
                "qb_execution": "frozen v6.3.6 rich-QB live field-read path",
                "qb_mobility": "frozen v6.3.6 pressure-response path",
                "wr_cb_matchup": "frozen v6.3.6 snap matchup path",
                "play_calling": "frozen v6.3.6 game_flow_policy + intent_ecology",
                "scoreboard": "game_loop_v13 event-derived only",
            },
            "v637_principle": (
                "Designed-rush participation is a finite role state. A nonzero positional "
                "rush prior does not mean every active WR/TE enters the carry tree in every game."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v637_runtime
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
    _record_v637_manifest(out)
    write_runtime_fingerprint_v637(out)


if __name__ == "__main__":
    main()
