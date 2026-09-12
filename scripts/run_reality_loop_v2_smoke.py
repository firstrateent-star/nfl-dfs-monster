from __future__ import annotations

import run_week1_v13_dispersion_test as runner

from monster.sim import matchup_kernel, resolution_ecology
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2


def configure_reality_loop_v2() -> None:
    """Activate the Reality Loop v2 causal seams without mutating the stable v1.3 runner."""

    # Historical outcome families provide the NFL center. Team/player/Madden matchup evidence
    # should explain deviations around that center, not re-apply average league difficulty.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25

    # Role/participation-aware snap ecology preserves individual skill while removing the old
    # selection bias where the strongest defensive players effectively participated in every
    # relevant interaction.
    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2


def main() -> None:
    configure_reality_loop_v2()
    runner.main()


if __name__ == "__main__":
    main()
