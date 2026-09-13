from __future__ import annotations

import numpy as np

from monster.sim.interaction_topology_v4 import pair_pass_rushers_to_blockers
from monster.sim.interaction_topology_v4b import (
    choose_coverage_participants,
    choose_pass_rushers,
    choose_protection_helper,
    choose_run_blockers,
    choose_run_participants,
    exposure_weighted_mean,
)
from monster.sim.snap_ecology import (
    CoverageAssignment,
    PassRushDuel,
    PassSnapResolution,
    RunSnapResolution,
    _defender_by_id,
    _edge,
    resolve_qb_read,
    team_profile_for_player,
)


def resolve_pass_protection(
    *,
    target_id: str,
    rushers: tuple[object, ...],
    base_pressure_rate: float,
    fallback_pass_protection: float,
    responsibility_key: str,
) -> tuple[float, float, float, float, str | None, tuple[PassRushDuel, ...]]:
    profile = team_profile_for_player(target_id)
    if profile is None or not profile.offensive_line or not rushers:
        pressure = float(
            np.clip(base_pressure_rate / max(fallback_pass_protection, 0.55), 0.08, 0.58)
        )
        return (
            pressure,
            float(np.clip(2.65 - 1.9 * (pressure - 0.20), 1.35, 3.25)),
            fallback_pass_protection,
            0.0,
            None,
            (),
        )

    rush = choose_pass_rushers(
        rushers,
        count=min(5, len(rushers)),
        responsibility_key=responsibility_key,
    )
    pairs = pair_pass_rushers_to_blockers(rush, profile.offensive_line)
    duels: list[PassRushDuel] = []
    for rusher, blocker in pairs:
        rusher_strength = float(getattr(rusher, "pass_rush", 1.0))
        blocker_strength = float(
            np.clip(
                0.70 * blocker.pass_block
                + 0.20 * blocker.awareness
                + 0.10 * blocker.stamina,
                0.65,
                1.35,
            )
        )
        rush_edge = _edge(rusher_strength, blocker_strength)
        duels.append(
            PassRushDuel(
                blocker.player_id,
                str(getattr(rusher, "player_id", "")) or None,
                blocker_strength,
                rusher_strength,
                rush_edge,
                float(np.clip(2.75 - 0.72 * rush_edge, 1.25, 3.55)),
            )
        )

    helper = choose_protection_helper(
        profile.protectors,
        responsibility_key=responsibility_key,
    )
    help_strength = 0.0 if helper is None else max(helper.pass_block - 0.80, 0.0)
    if duels and help_strength > 0.0:
        idx = max(range(len(duels)), key=lambda i: duels[i].rush_edge)
        duel = duels[idx]
        helped = float(np.clip(duel.rush_edge - 0.32 * help_strength, -1.0, 1.0))
        duels[idx] = PassRushDuel(
            duel.blocker_id,
            duel.rusher_id,
            duel.blocker_strength,
            duel.rusher_strength,
            helped,
            float(np.clip(duel.time_to_pressure + 0.30 * help_strength, 1.25, 3.70)),
        )

    worst = max(duel.rush_edge for duel in duels)
    mean = float(np.mean([duel.rush_edge for duel in duels]))
    pocket = float(
        np.clip(
            1.0 - 0.20 * worst - 0.09 * mean + 0.06 * help_strength,
            0.68,
            1.30,
        )
    )
    pressure = float(
        np.clip(
            base_pressure_rate
            * (1.0 + 0.34 * worst + 0.18 * mean)
            / max(pocket, 0.65),
            0.07,
            0.62,
        )
    )
    fastest = min(duels, key=lambda duel: duel.time_to_pressure)
    return (
        pressure,
        fastest.time_to_pressure,
        pocket,
        help_strength,
        fastest.rusher_id,
        tuple(duels),
    )


def resolve_coverage_assignment(
    *,
    target: object,
    defenders: tuple[object, ...],
    responsibility_key: str,
) -> CoverageAssignment:
    target_id = str(getattr(target, "player_id", ""))
    if not defenders:
        return CoverageAssignment(None, target_id, 1.0, 0.0, 0.0, 0.0, 0.0)

    participants = choose_coverage_participants(
        target=target,
        defenders=defenders,
        responsibility_key=responsibility_key,
    )
    primary = _defender_by_id(defenders, participants.primary_defender_id)
    safety = _defender_by_id(defenders, participants.safety_defender_id)
    bracket = _defender_by_id(defenders, participants.bracket_defender_id)

    local = 1.0 if primary is None else float(getattr(primary, "coverage", 1.0))
    separation = _edge(float(getattr(target, "efficiency", 1.0)), local)
    safety_help = (
        0.0
        if safety is None
        else float(np.clip(float(getattr(safety, "coverage", 1.0)) - 0.82, 0.0, 0.40))
    )
    bracket_factor = (
        0.0
        if bracket is None
        else float(
            np.clip(
                (float(getattr(bracket, "coverage", 1.0)) - 0.85) * 0.45,
                0.0,
                0.22,
            )
        )
    )
    zone_overlap = float(
        np.clip(exposure_weighted_mean(defenders, "coverage") - 0.92, 0.0, 0.22)
    )
    return CoverageAssignment(
        participants.primary_defender_id,
        target_id,
        local,
        separation,
        safety_help,
        bracket_factor,
        zone_overlap,
        participants.safety_defender_id,
        participants.bracket_defender_id,
    )


def resolve_pass_snap(
    *,
    target: object,
    defense: object,
    pass_protection: float,
    quarterback_efficiency: float,
    responsibility_key: str,
) -> PassSnapResolution:
    coverage = resolve_coverage_assignment(
        target=target,
        defenders=tuple(getattr(defense, "coverage", ())),
        responsibility_key=responsibility_key,
    )
    pressure, ttp, pocket, help_strength, rusher_id, _ = resolve_pass_protection(
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
    effective_coverage = float(
        np.clip(
            coverage.local_coverage
            + 0.50 * coverage.safety_help
            + 0.70 * coverage.bracket_factor
            + 0.45 * coverage.zone_overlap,
            0.55,
            1.60,
        )
    )
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


def resolve_run_snap(
    *,
    rusher: object,
    defense: object,
    run_blocking: float,
    responsibility_key: str,
    run_geometry: str | None,
) -> RunSnapResolution:
    profile = team_profile_for_player(str(getattr(rusher, "player_id", "")))
    if profile is None:
        selected_blockers: tuple[object, ...] = ()
    else:
        selected_blockers = choose_run_blockers(
            profile.offensive_line,
            profile.protectors,
            run_geometry=run_geometry,
        )

    if selected_blockers:
        weights = np.asarray(
            [max(float(getattr(blocker, "snap_weight", 0.0)), 0.001) for blocker in selected_blockers],
            dtype=float,
        )
        lane_blocking = float(
            np.average(
                np.asarray(
                    [float(getattr(blocker, "run_block", 1.0)) for blocker in selected_blockers],
                    dtype=float,
                ),
                weights=weights,
            )
        )
    else:
        lane_blocking = run_blocking

    front = tuple(getattr(defense, "front", ()))
    coverage = tuple(getattr(defense, "coverage", ()))
    participants = choose_run_participants(
        rusher=rusher,
        front=front,
        coverage=coverage,
        responsibility_key=responsibility_key,
        run_geometry=run_geometry,
    )
    primary = _defender_by_id(front, participants.box_defender_id)
    pursuit_pool = coverage if coverage else front
    pursuit = _defender_by_id(pursuit_pool, participants.pursuit_defender_id)

    unit_front_fit = exposure_weighted_mean(front, "run_defense") if front else 1.0
    local_front_fit = 1.0 if primary is None else float(getattr(primary, "run_defense", 1.0))
    front_fit = float(np.clip(0.72 * unit_front_fit + 0.28 * local_front_fit, 0.55, 1.55))

    unit_second_level = exposure_weighted_mean(pursuit_pool, "tackling") if pursuit_pool else 1.0
    local_second_level = 1.0 if pursuit is None else float(getattr(pursuit, "tackling", 1.0))
    second_level = float(np.clip(0.58 * unit_second_level + 0.42 * local_second_level, 0.55, 1.55))

    runner_skill = float(getattr(rusher, "efficiency", 1.0))
    runner_edge = _edge(runner_skill * lane_blocking, front_fit)
    stuff = float(
        np.clip(
            float(getattr(defense, "run_stuff_rate", 0.18))
            * front_fit
            / max(lane_blocking, 0.55)
            * (1.0 - 0.14 * runner_edge),
            0.04,
            0.50,
        )
    )
    yards = float(
        np.clip(
            runner_skill
            * lane_blocking
            / max(front_fit, 0.60)
            * (1.0 + 0.10 * runner_edge)
            / max(second_level**0.16, 0.90),
            0.46,
            1.78,
        )
    )
    return RunSnapResolution(
        lane_blocking,
        front_fit,
        second_level,
        runner_edge,
        stuff,
        yards,
        participants.box_defender_id,
        participants.pursuit_defender_id,
    )
