from __future__ import annotations

import json
from datetime import date

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63

from monster.sim.current_role_guard_v63 import install_current_role_guard


MATCHUP = (("DEN", "KC"),)
GAME_DATE = date(2026, 9, 14)


def configure_v63_current_role_truth() -> None:
    """Keep v6.3 physics intact while preserving active current starters in role worlds."""
    v63.configure_reality_loop_v63()
    integrated = v61.runner.integrated
    native_compile = integrated.compile_current_skill_pools

    def compile_with_current_role_truth(personnel, historical_usage, **kwargs):
        install_current_role_guard(personnel)
        return native_compile(personnel, historical_usage, **kwargs)

    integrated.compile_current_skill_pools = compile_with_current_role_truth


def main() -> None:
    integrated = v61.runner.integrated
    original_matchups = integrated.MATCHUPS
    original_game_date = integrated.GAME_DATE
    original_configure = v61.configure_reality_loop_v2
    try:
        integrated.MATCHUPS = MATCHUP
        integrated.GAME_DATE = GAME_DATE
        v61.configure_reality_loop_v2 = configure_v63_current_role_truth
        v61.main()
    finally:
        integrated.MATCHUPS = original_matchups
        integrated.GAME_DATE = original_game_date
        v61.configure_reality_loop_v2 = original_configure

    out = v61._first_out_path()
    v63._TARGET_ROLE_TELEMETRY.write(out)
    v63._record_v63_manifest(out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "current_role_truth_experiment_active": True,
                "current_starter_receiving_guard_active": True,
                "receiver_candidate_limit": 10,
                "target_role_participant_limit": 8,
                "principle_current_role_truth": (
                    "A healthy current depth starter remains eligible in the route/opportunity universe. "
                    "Historical usage shapes relative target probability but cannot erase current role."
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
