from __future__ import annotations

from dataclasses import replace

import numpy as np

from monster.sim.interaction_topology_v4b import choose_run_blockers, exposure_weighted_mean
from monster.sim.reality_snap_v5 import (
    coverage_intent_adjustments,
    planned_rushers,
    pressure_plan_multiplier,
    record_pass_resolution,
    record_run_resolution,
    record_trench_duels,
    run_fit_multiplier,
)
from monster.sim.snap_ecology import (
    CoverageAssignment,
    PassSnapResolution,
    RunSnapResolution,
    _defender_by_id,
    _edge,
    resolve_qb_read,
    team_profile_for_player,
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
    """Reality Loop coverage seam with v5 shell intent and v4B player responsibility."""
    coverage = resolve_coverage_assignment_v4b(
        target=target,
        defenders=defenders,
        responsibility_key=responsibility_key,
    )

    # Legacy causal tests call the seam without a real snap world. Preserve their important
    # invariant: an outside receiver should not draw a LB/S as primary merely because that
    # defender has the highest rating. Production v5 calls always carry a non-static key.
    if responsibility_key == "static" and str(getattr(target, "position", "")).upper() == "WR":
        corners = tuple(
            defender
            for defender in defenders
            if str(getattr(defender, "position", "")).upper() in {"CB", "DB"}
        )
        if corners:
            primary = max(
                corners,
                key=lambda defender: (
                    float(getattr(defender, "snap_weight", 0.0) or 0.0),
                    str(getattr(defender, "player_id", "")),
                ),
            )
            local = float(getattr(primary, "coverage", 1.0))
            coverage = replace(
                coverage,
                defender_id=str(getattr(primary, "player_id", "")) or None,
                local_coverage=local,
                separation_edge=_edge(float(getattr(target, "efficiency", 1.0)), local),
            )

    safety_mult, bracket_mult, zone_mult = coverage_intent_adjustments(responsibility_key)
    return replace(
        coverage,
        safety_help=float(np.clip(coverage.safety_help * safety_mult, 0.0, 0.45)),
        bracket_factor=float(np.clip(coverage.bracket_factor * bracket_mult, 0.0, 0.28)),
        zone_overlap=float(np.clip(coverage.zone_overlap * zone_mult, 0.0, 0.28)),
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
    front = tuple(getattr(defense, "front", ()))
    rushers = planned_rushers(responsibility_key, front)
    pressure, ttp, pocket, help_strength, rusher_id, duels = resolve_pass_protection_v4b(
        target_id=str(getattr(target, "player_id", "")),
        rushers=rushers,
        base_pressure_rate=float(getattr(defense, "pressure_rate", 0.297832)),
        fallback_pass_protection=pass_protection,
        responsibility_key=responsibility_key,
    )
    pressure = float(np.clip(pressure * pressure_plan_multiplier(responsibility_key), 0.07, 0.62))
    qb_read = resolve_qb_read(
        quarterback_efficiency=quarterback_efficiency,
        separation_edge=coverage.separation_edge,
        time_to_pressure=ttp,
        safety_help=coverage.safety_help,
        bracket_factor=coverage.bracket_factor,
        zone_overlap=coverage.zone_overlap,
    )

    effective_coverage = float(np.clip(coverage.local_coverage, 0.65, 1.40))
    result = PassSnapResolution(
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
    target_id = str(getattr(target, "player_id", ""))
    record_pass_resolution(
        responsibility_key,
        target_id,
        primary_defender_id=result.primary_defender_id,
        primary_rusher_id=result.primary_rusher_id,
        safety_defender_id=result.safety_defender_id,
        bracket_defender_id=result.bracket_defender_id,
        pressure_probability=result.pressure_probability,
        time_to_pressure=result.time_to_pressure,
        coverage_strength=result.coverage_strength,
    )
    record_trench_duels(responsibility_key, duels)
    return result


def resolve_run_snap_v2(
    *,
    rusher: object,
    defense: object,
    run_blocking: float,
    responsibility_key: str = "static",
    run_geometry: str | None = None,
) -> RunSnapResolution:
    """V5 run seam: geometry-specific blockers plus stronger local DL/LB/S jurisdiction."""
    base = resolve_run_snap_v4b(
        rusher=rusher,
        defense=defense,
        run_blocking=run_blocking,
        responsibility_key=responsibility_key,
        run_geometry=run_geometry,
    )
    front = tuple(getattr(defense, "front", ()))
    coverage = tuple(getattr(defense, "coverage", ()))
    primary = _defender_by_id(front, base.primary_defender_id)
    pursuit_pool = coverage if coverage else front
    pursuit = _defender_by_id(pursuit_pool, base.pursuit_defender_id)

    unit_front = exposure_weighted_mean(front, "run_defense") if front else 1.0
    local_front = 1.0 if primary is None else float(getattr(primary, "run_defense", 1.0))
    front_fit = float(
        np.clip(
            (0.50 * unit_front + 0.50 * local_front) * run_fit_multiplier(responsibility_key),
            0.55,
            1.55,
        )
    )

    unit_second = exposure_weighted_mean(pursuit_pool, "tackling") if pursuit_pool else 1.0
    local_second = 1.0 if pursuit is None else float(getattr(pursuit, "tackling", 1.0))
    second_level = float(np.clip(0.45 * unit_second + 0.55 * local_second, 0.55, 1.55))

    runner_skill = float(getattr(rusher, "efficiency", 1.0))
    runner_edge = _edge(runner_skill * base.lane_blocking, front_fit)
    stuff = float(
        np.clip(
            float(getattr(defense, "run_stuff_rate", 0.18))
            * front_fit
            / max(base.lane_blocking, 0.55)
            * (1.0 - 0.14 * runner_edge),
            0.04,
            0.50,
        )
    )
    yards = float(
        np.clip(
            runner_skill
            * base.lane_blocking
            / max(front_fit, 0.60)
            * (1.0 + 0.10 * runner_edge)
            / max(second_level**0.16, 0.90),
            0.46,
            1.78,
        )
    )
    result = RunSnapResolution(
        base.lane_blocking,
        front_fit,
        second_level,
        runner_edge,
        stuff,
        yards,
        base.primary_defender_id,
        base.pursuit_defender_id,
    )

    profile = team_profile_for_player(str(getattr(rusher, "player_id", "")))
    blockers: tuple[object, ...] = ()
    if profile is not None:
        blockers = choose_run_blockers(
            profile.offensive_line,
            profile.protectors,
            run_geometry=run_geometry,
        )
    record_run_resolution(
        responsibility_key,
        primary_defender_id=result.primary_defender_id,
        pursuit_defender_id=result.pursuit_defender_id,
        run_blocker_ids=tuple(str(getattr(blocker, "player_id", "")) for blocker in blockers),
        lane_blocking=result.lane_blocking,
        front_fit=result.front_fit,
        second_level_fit=result.second_level_fit,
    )
    return result
