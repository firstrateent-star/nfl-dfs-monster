from __future__ import annotations

import run_week1_v13_dispersion_test as runner

from monster.sim import matchup_kernel, play_kernel, resolution_ecology
from monster.sim.clock_ecology_v2 import sample_snap_cadence_v2
from monster.sim.resolution_bands_v2 import resolve_run_contact_v2, resolve_run_ecology_v2
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

    # Preserve designed-run failure/explosive branches while restoring the empirical 3+/5+
    # routine bands, and give QB scrambles their own open-field 10-14 yard branch.
    resolution_ecology.resolve_run_ecology = resolve_run_ecology_v2
    play_kernel.resolve_run_contact = resolve_run_contact_v2

    # Drive conversion/survival is already close to NFL reality; low play volume was therefore
    # a clock ecology problem rather than an invitation to inflate offensive success.
    play_kernel._sample_snap_cadence = sample_snap_cadence_v2


def main() -> None:
    configure_reality_loop_v2()
    runner.main()


if __name__ == "__main__":
    main()
