from __future__ import annotations

import audit_play_gain_reality as audit

from monster.sim import matchup_kernel, resolution_ecology
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2


def main() -> None:
    """Run the NFL-vs-Monster play-family audit on the Reality Loop v2 identity path."""

    # Historical depth/run outcomes already contain average NFL matchup difficulty. Rich
    # player-v-player identity therefore perturbs those priors rather than charging the same
    # average defensive difficulty again. The v2 snap ecology also removes the structural
    # selection bias where the defense's best coverage/front players were effectively active
    # on every snap regardless of role/participation.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2
    audit._team_identity = enhanced_team_identity
    audit._defensive_unit = enhanced_defensive_unit
    audit.main()


if __name__ == "__main__":
    main()
