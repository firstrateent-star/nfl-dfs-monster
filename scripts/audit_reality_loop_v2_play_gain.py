from __future__ import annotations

import audit_play_gain_reality as audit

from monster.sim import resolution_ecology
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity


def main() -> None:
    """Run the NFL-vs-Monster play-family audit on the Reality Loop v2 identity path."""

    # Historical depth/run outcomes already contain average NFL matchup difficulty. The rich
    # identity layer now supplies direct player-v-player evidence, so the older coarse aggregate
    # matchup bridge should perturb those priors rather than charging the same difficulty again.
    # Pass keeps moderate authority; run authority is lower because individual OL/front/runner
    # channels now act inside the snap and the prior 0.50 authority over-produced negative runs.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    audit._team_identity = enhanced_team_identity
    audit._defensive_unit = enhanced_defensive_unit
    audit.main()


if __name__ == "__main__":
    main()
