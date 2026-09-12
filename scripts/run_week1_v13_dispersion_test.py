from __future__ import annotations

import json
import sys
from pathlib import Path

import run_week1_v13_integrated as integrated
from monster.sim import resolution_ecology
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity


def _argument_path(flag: str, default: str) -> Path:
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path(default)


def _record_experiment() -> None:
    first_out = _argument_path("--first-out", "artifacts/week1-v13-first-sim")
    manifest_path = first_out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "dispersion_architecture_test": True,
            "rich_player_capability_bridge_active": True,
            "team_offense_multidimensional_identity_active": True,
            "team_defense_context_active": True,
            "primary_receiver_rotation_active": True,
            "primary_receiver_rotation_max_players": 6,
            "pass_matchup_relative_authority": resolution_ecology._COARSE_MATCHUP_AUTHORITY,
            "score_dispersion_not_directly_calibrated": True,
            "principle_dispersion": (
                "Totals and margins must emerge from offense, defense, personnel, role and matchup "
                "differences; no target game scores or sportsbook inputs are used."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    # The empirical depth prior remains the center, but the now-richer player-v-player graph
    # earns more relative authority than the previous coarse bridge. 0.50 is still explicitly
    # shrunk and cannot replace the historical depth outcome prior.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.50
    integrated._team_identity = enhanced_team_identity
    integrated._defensive_unit = enhanced_defensive_unit
    integrated.main()
    _record_experiment()


if __name__ == "__main__":
    main()
