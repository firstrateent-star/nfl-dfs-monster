from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63
import run_reality_loop_v634 as v634

from monster.sim import resolution_ecology
from monster.sim.current_role_guard_v635 import install_current_role_guard_v635


_NATIVE_APPLY_V634 = v634.apply_matchup_identity_authority_v634


def _apply_identity_authority_v635(identity, **kwargs):
    """Keep rich players/unit identity, but remove the dead team-pass proxy authority.

    Normal pass snaps are now driven by the actual rich QB execution channel in play_kernel.
    Leaving the v6.3.4 team pass_efficiency multiplier active would only affect a few
    downstream scoring seams (not ordinary throws), creating false confidence and double
    authority. Rushing team context remains live because rush_efficiency is consumed on
    every run.
    """

    enhanced, trace = _NATIVE_APPLY_V634(identity, **kwargs)
    return (
        replace(enhanced, pass_efficiency=identity.pass_efficiency),
        replace(trace, pass_efficiency_after=identity.pass_efficiency),
    )


def configure_reality_loop_v635() -> None:
    """Compose v6.3.5 identity propagation + role separation over v6.3 physics."""

    # v6.3.4 closures resolve these module globals at runtime. Replace only the two seams
    # proven by the paired Week-1 audit to be wrong/incomplete.
    v634.install_current_role_guard = install_current_role_guard_v635
    v634.apply_matchup_identity_authority_v634 = _apply_identity_authority_v635
    v634.configure_reality_loop_v634()

    # Do not claim the legacy _COARSE_* variables as production authority. The live depth/run
    # ecology is governed by _SNAP_MATCHUP_AUTHORITY / _SNAP_RUN_MATCHUP_AUTHORITY.
    v61.runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = None


def _record_v635_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v635_shadow_active": True,
            "v635_identity_propagation_active": True,
            "v635_role_participation_share_separation_active": True,
            "v635_current_rushing_role_truth_active": True,
            "v635_sparse_target_reserve_only_active": True,
            "v635_rich_qb_execution_to_live_read_active": True,
            "v635_rich_qb_mobility_to_pressure_response_active": True,
            "v635_live_matchup_to_yac_propagation_active": True,
            "v635_dead_team_pass_proxy_removed": True,
            "v635_live_pass_matchup_authority": float(
                resolution_ecology._SNAP_MATCHUP_AUTHORITY
            ),
            "v635_live_run_matchup_authority": float(
                resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY
            ),
            "v635_stale_coarse_matchup_knob_is_authority": False,
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "runtime_authority_map_v635": {
                "participation_truth": "current_role_guard_v635",
                "conditional_target_share": "historical target hierarchy + sparse current reserve",
                "conditional_rush_share": "v6.3 finite role core + current depth/snap reweight",
                "qb_execution": "RichPlayerIdentity.qb_execution_skill -> field read/QB response",
                "qb_mobility": "RichPlayerIdentity.mobility_skill -> pressure response",
                "wr_cb_matchup": "snap_ecology_v4b + matchup_kernel",
                "post_catch_matchup_space": "bounded matchup yards multiplier -> YAC ecology",
                "run_matchup": "snap_ecology_v4b + matchup_kernel + live rush_efficiency",
                "play_calling": "game_flow_policy + intent_ecology",
                "persistent_environment": "reality_v63",
                "scoreboard": "game_loop_v13 event-derived only",
            },
            "v635_principle": (
                "Current truth decides who can participate; historical/current role evidence "
                "decides conditional opportunity; rich player identity must survive through "
                "the actual event mechanism that resolves the play."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = configure_reality_loop_v635
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original

    out = v61._first_out_path()
    v63._TARGET_ROLE_TELEMETRY.write(out)
    v634._IDENTITY_TELEMETRY.write(out)
    v63._record_v63_manifest(out)
    v634._record_v634_manifest(out)
    _record_v635_manifest(out)


if __name__ == "__main__":
    main()
