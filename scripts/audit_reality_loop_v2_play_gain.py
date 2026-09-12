from __future__ import annotations

import audit_play_gain_reality as audit

from monster.sim import resolution_ecology
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity


def main() -> None:
    """Run the existing NFL-vs-Monster play-family audit on the Reality Loop v2 identity path."""

    # Match the live dispersion experiment: historical outcomes are the center while richer
    # player-v-player identity gets bounded relative authority.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.50
    audit._team_identity = enhanced_team_identity
    audit._defensive_unit = enhanced_defensive_unit
    audit.main()


if __name__ == "__main__":
    main()
