from __future__ import annotations

import json
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634
import run_reality_loop_v635 as v635
import run_reality_loop_v636 as v636
from runtime_v639_composer import (
    compose_v639_runtime,
    write_runtime_fingerprint_v639,
)

from monster.sim import resolution_ecology


def _record_v639_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v639_shadow_active": True,
            "v636_wr_te_gadget_rush_mass_preserved_from_native_plan": False,
            "v639_gadget_rush_decomposition_active": True,\n            "v639_oos_recurrence_calibration_active": True,
            "v639_rotation_exposure_separate_from_rush_entry": True,
            "v639_heuristic_rush_role_probability_is_not_entry_authority": True,
            "v639_conditional_carries_from_2022_2025_empirical_distribution": True,
            "v639_qb_rush_mass_frozen_to_v636": True,
            "v639_rb_internal_hierarchy_frozen_to_v636": True,
            "v639_target_sampler_frozen_to_v635": True,
            "v639_week1_truth_used_for_priors": False,
            "v639_empirical_reference_team_gadget_share_mean_2022_2025": 0.03674310517636209,
            "v639_empirical_reference_wr_entry_rate_2022_2025": 0.1315333672949567,
            "v639_empirical_reference_te_entry_rate_2022_2025": 0.03567787971457696,
            "v639_live_pass_matchup_authority": float(
                resolution_ecology._SNAP_MATCHUP_AUTHORITY
            ),
            "v639_live_run_matchup_authority": float(
                resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY
            ),
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "runtime_authority_map_v639": {
                "participation_truth": "current roster/depth/snap truth inherited from v6.3.6",
                "conditional_target_share": "frozen v6.3.5 target sampler",
                "qb_rush_mass": "frozen v6.3.6 rushing plan",
                "wr_te_rotation_exposure": "current depth + conditional offense snap truth",
                "wr_te_rush_entry": "2022-2025 next-season recurrence by raw prior-season carries",
                "wr_te_conditional_carries": "2022-2025 market-blind empirical position PMF",
                "conditional_rb_rush_share": (
                    "frozen v6.3.6 RB hierarchy after gadget carry mass is reserved"
                ),
                "qb_execution": "frozen v6.3.6 rich-QB live field-read path",
                "qb_mobility": "frozen v6.3.6 pressure-response path",
                "wr_cb_matchup": "frozen v6.3.6 snap matchup path",
                "play_calling": "frozen v6.3.6 game_flow_policy + intent_ecology",
                "scoreboard": "game_loop_v13 event-derived only",
            },
            "v639_principle": (
                "A gadget role is a finite football state: a player must be in the "
                "offensive rotation, then enter the designed-rush tree, then receive "
                "a discrete conditional carry workload. Entry probability is not "
                "reused as carry-share authority."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v639_runtime
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
    _record_v639_manifest(out)
    write_runtime_fingerprint_v639(out)


if __name__ == "__main__":
    main()
