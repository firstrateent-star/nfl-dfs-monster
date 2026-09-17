from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl
import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63

from monster.feature_compile.v634_matchup_identity_authority import (
    DEFAULT_V634_AUTHORITY,
    MatchupIdentityTraceV634,
    apply_matchup_identity_authority_v634,
)
from monster.sim import resolution_ecology
from monster.sim.current_role_guard_v63 import install_current_role_guard


class IdentityAuthorityTelemetryV634:
    def __init__(self) -> None:
        self._by_team: dict[str, MatchupIdentityTraceV634] = {}

    def register(self, trace: MatchupIdentityTraceV634) -> None:
        self._by_team[trace.team_id] = trace

    def write(self, out: Path) -> None:
        if not self._by_team:
            return
        out.mkdir(parents=True, exist_ok=True)
        rows = [asdict(self._by_team[team]) for team in sorted(self._by_team)]
        pl.DataFrame(rows).write_csv(out / "matchup_identity_trace_v634.csv")


_IDENTITY_TELEMETRY = IdentityAuthorityTelemetryV634()


def configure_reality_loop_v634() -> None:
    """Compose v6.3 with current-role truth and stronger causal identity authority.

    This remains a shadow layer. It changes player/team football mechanisms only; it does not
    sample score, fantasy points, or market information.
    """

    v63.configure_reality_loop_v63()

    integrated = v61.runner.integrated
    native_compile_pools = integrated.compile_current_skill_pools
    native_team_identity = v61.runner.enhanced_team_identity

    def compile_current_skill_pools_v634(personnel, historical_usage, **kwargs):
        # Current depth/availability protects who can participate. Historical usage still owns
        # relative opportunity authority; the guard is not allowed to flatten target hierarchy.
        install_current_role_guard(personnel)
        return native_compile_pools(personnel, historical_usage, **kwargs)

    def enhanced_team_identity_v634(*args, **kwargs):
        identity = native_team_identity(*args, **kwargs)
        pool = args[1] if len(args) > 1 else kwargs["pool"]
        reality = args[2] if len(args) > 2 else kwargs["reality"]
        unit_players = args[3] if len(args) > 3 else kwargs.get("unit_players", ())
        state = args[4] if len(args) > 4 else kwargs["state"]
        enhanced, trace = apply_matchup_identity_authority_v634(
            identity,
            pool=pool,
            reality=reality,
            unit_players=tuple(unit_players),
            state=state,
            config=DEFAULT_V634_AUTHORITY,
        )
        _IDENTITY_TELEMETRY.register(trace)
        return enhanced

    integrated.compile_current_skill_pools = compile_current_skill_pools_v634
    v61.runner.enhanced_team_identity = enhanced_team_identity_v634

    # These are relative football-mechanism authorities, not point multipliers. v6.1 uses
    # 0.35 pass / 0.25 run. The shadow increases them cautiously while keeping full 1v1
    # assignment and safety/help context intact.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = (
        DEFAULT_V634_AUTHORITY.pass_matchup_relative_authority
    )
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = (
        DEFAULT_V634_AUTHORITY.run_matchup_relative_authority
    )
    v61.runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = (
        DEFAULT_V634_AUTHORITY.pass_matchup_relative_authority
    )


def _record_v634_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v634_shadow_active": True,
            "v634_matchup_identity_authority_active": True,
            "v634_current_role_truth_guard_active": True,
            "v634_small_rotation_role_cap_repair_active": True,
            "v634_identity_trace": "matchup_identity_trace_v634.csv",
            "v634_qb_to_pass_efficiency_authority": (
                DEFAULT_V634_AUTHORITY.qb_to_pass_efficiency
            ),
            "v634_runner_to_rush_efficiency_authority": (
                DEFAULT_V634_AUTHORITY.runner_to_rush_efficiency
            ),
            "v634_pass_matchup_relative_authority": (
                DEFAULT_V634_AUTHORITY.pass_matchup_relative_authority
            ),
            "v634_run_matchup_relative_authority": (
                DEFAULT_V634_AUTHORITY.run_matchup_relative_authority
            ),
            "direct_score_adjustment": False,
            "direct_fantasy_adjustment": False,
            "market_inputs_used_for_football": False,
            "promotion_status": "SHADOW_NOT_PROMOTED",
            "v634_principle": (
                "Randomness chooses among plausible football worlds; current player, unit, "
                "scheme and matchup identity determine which worlds are plausible. Current "
                "availability protects participation without equalizing historical opportunity."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = configure_reality_loop_v634
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original

    out = v61._first_out_path()
    v63._TARGET_ROLE_TELEMETRY.write(out)
    _IDENTITY_TELEMETRY.write(out)
    v63._record_v63_manifest(out)
    _record_v634_manifest(out)


if __name__ == "__main__":
    main()
