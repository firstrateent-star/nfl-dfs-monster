from __future__ import annotations

import audit_week1_v13_drive_survival as audit

from monster.sim import matchup_kernel, resolution_ecology
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2


def main() -> None:
    """Audit the active Reality Loop v2 game against historical NFL drive anatomy."""

    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2
    audit.main()


if __name__ == "__main__":
    main()
