from __future__ import annotations

import numpy as np

from monster.sim.snap_ecology import (
    CoverageAssignment,
    PassSnapResolution,
    RunSnapResolution,
    resolve_qb_read,
)
from monster.sim.snap_ecology_v4b import (
    resolve_coverage_assignment as resolve_coverage_assignment_v4b,
    resolve_pass_protection as resolve_pass_protection_v4b,
    resolve_run_snap as resolve_run_snap_v4b,
)


def resolve_coverage_assignment_v2(
    *,
    target: object,
    defenders: tuple[object, ...],
    responsibility_key: str = "static",
) -> CoverageAssignment:
    """Reality Loop coverage seam with v4B responsibility and v2 calibrated authority.

    Participant identity is selected by v4B from role + snap exposure only. The selected
    defender's actual coverage skill then resolves the local duel. Safety/bracket/zone help
    remain separate causal channels so the same defensive difficulty is not charged twice.
    """
    return resolve_coverage_assignment_v4b(
        target=target,
        defenders=defenders,
        responsibility_key=responsibility_key,
    )


def resolve_pass_snap_v2(
    *,
    target: object,
    defense: object,
    pass_protection: float,
    quarterback_efficiency: float,
    responsibility_key: str = "static",
) -> PassSnapResolution:
    defenders = tuple(getattr(defense, "coverage", ()))
    coverage = resolve_coverage_assignment_v2(
        target=target,
        defenders=defenders,
        responsibility_key=responsibility_key,
    )
    pressure, ttp, pocket, help_strength, rusher_id, _ = resolve_pass_protection_v4b(
        target_id=str(getattr(target, "player_id", "")),
        rushers=tuple(getattr(defense, "front", ())),
        base_pressure_rate=float(getattr(defense, "pressure_rate", 0.297832)),
        fallback_pass_protection=pass_protection,
        responsibility_key=responsibility_key,
    )
    qb_read = resolve_qb_read(
        quarterback_efficiency=quarterback_efficiency,
        separation_edge=coverage.separation_edge,
        time_to_pressure=ttp,
        safety_help=coverage.safety_help,
        bracket_factor=coverage.bracket_factor,
        zone_overlap=coverage.zone_overlap,
    )

    # Preserve the Reality Loop v2 authority seam: safety/bracket/zone help remain explicit
    # channels rather than being buried inside coverage strength and charged twice downstream.
    effective_coverage = float(np.clip(coverage.local_coverage, 0.65, 1.40))
    return PassSnapResolution(
        pressure,
        ttp,
        pocket,
        help_strength,
        effective_coverage,
        coverage.separation_edge,
        coverage.safety_help,
        coverage.bracket_factor,
        coverage.zone_overlap,
        coverage.defender_id,
        rusher_id,
        qb_read,
        coverage.safety_defender_id,
        coverage.bracket_defender_id,
    )


def resolve_run_snap_v2(
    *,
    rusher: object,
    defense: object,
    run_blocking: float,
    responsibility_key: str = "static",
    run_geometry: str | None = None,
) -> RunSnapResolution:
    """Preserve the v2 run-resolution seam while promoting v4B run topology.

    Historical run priors and the progressive v2/v3 tail remain downstream in
    ``resolution_ecology``. This function now owns only geometry-specific blocking, box fit,
    pursuit responsibility and their local skill interactions.
    """
    return resolve_run_snap_v4b(
        rusher=rusher,
        defense=defense,
        run_blocking=run_blocking,
        responsibility_key=responsibility_key,
        run_geometry=run_geometry,
    )
