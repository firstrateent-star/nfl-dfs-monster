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
from runtime_v720_composer import (
    compose_v720_runtime,
    write_runtime_fingerprint_v720,
)


def _record_v720_manifest(out) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_engine_v720_shadow_active": True,
            "v720_base_architecture": "v7.0.1-shadow-qb-rush-family-authority",
            "v720_current_depth_seat_authority_active": True,
            "v720_exact_ol_seats_active": True,
            "v720_current_role_snap_participation_active": True,
            "v720_designed_qb_run_entry_probability_active": True,
            "v720_scramble_runtime_changed": False,
            "v720_receiver_alignment_route_topology_active": True,
            "v720_post_participation_usage_share_reintroduced": False,
            "v720_defensive_attribution_uses_realized_snap_topology": True,
            "v720_live_player_mutation_active": True,
            "v720_qb_backup_promotion_active": True,
            "v720_special_teams_player_identity_active": True,
            "v720_special_teams_player_telemetry": (
                "same_world_special_teams_players_v72.csv"
            ),
            "v720_live_mutation_telemetry": "live_mutations_v72.csv",
            "v720_week1_truth_used_for_priors": False,
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "runtime_authority_map_v720": {
                "pregame_availability": "health state + roster status",
                "current_participation": (
                    "current depth seat/rank + conditional snap evidence"
                ),
                "offensive_line": (
                    "LT/LG/C/RG/RT current seat ownership before historical snap weight"
                ),
                "designed_qb_runs": (
                    "team/QB run-geometry evidence converted to concept-entry probability"
                ),
                "scrambles": "unchanged V7.1 dropback-response mechanism",
                "receiver_opportunity": (
                    "live participant -> route/depth family -> alignment -> matchup/QB read"
                ),
                "defensive_package": (
                    "current-role-weighted live participants inside sampled package"
                ),
                "defensive_box_attribution": (
                    "realized rusher/coverage/pursuit assignment from same snap"
                ),
                "live_mutation": (
                    "deterministic active/limited/out transition with backup substitution"
                ),
                "special_teams": (
                    "current K/P/returner identity + bounded specialist capability"
                ),
                "scoreboard": "game_loop_v13 event-derived only",
            },
            "v720_principle": (
                "League ecology is not tuned to Week 1. V7.2 repairs the allocation "
                "of football reality to current participants and causal identities."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = compose_v720_runtime
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
    _record_v720_manifest(out)
    write_runtime_fingerprint_v720(out)


if __name__ == "__main__":
    main()
