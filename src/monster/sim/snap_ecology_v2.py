from __future__ import annotations

import numpy as np

from monster.sim.snap_ecology import (
    CoverageAssignment,
    PassSnapResolution,
    RunSnapResolution,
    _edge,
    resolve_pass_protection,
    resolve_qb_read,
    team_profile_for_player,
)


def _weighted_mean(items: tuple[object, ...], attribute: str, default: float = 1.0) -> float:
    if not items:
        return default
    values = np.asarray([float(getattr(item, attribute, default)) for item in items], dtype=float)
    weights = np.asarray(
        [max(float(getattr(item, "snap_weight", 1.0)), 0.01) for item in items],
        dtype=float,
    )
    return float(np.average(values, weights=weights))


def _coverage_pool(target: object, defenders: tuple[object, ...]) -> tuple[object, ...]:
    """Return defenders with plausible first-order responsibility for this target role.

    The original snap ecology ranked the entire defense by coverage skill and therefore put
    the best coverage player on essentially every target. Historical pass outcomes already
    include ordinary NFL coverage difficulty, so that selection rule double-charged defense
    and erased player/team offensive identity. We preserve Madden/defender skill but make
    assignment role-aware and participation-aware.
    """

    position = str(getattr(target, "position", "")).upper()
    if position == "WR":
        preferred = {"CB", "DB", "S", "FS", "SS"}
    elif position == "TE":
        preferred = {"LB", "ILB", "OLB", "MLB", "S", "FS", "SS", "DB"}
    elif position in {"RB", "FB"}:
        preferred = {"LB", "ILB", "OLB", "MLB", "S", "FS", "SS", "DB", "CB"}
    else:
        preferred = {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB"}
    pool = tuple(d for d in defenders if str(getattr(d, "position", "")).upper() in preferred)
    return pool or defenders


def resolve_coverage_assignment_v2(
    *, target: object, defenders: tuple[object, ...]
) -> CoverageAssignment:
    target_id = str(getattr(target, "player_id", ""))
    if not defenders:
        return CoverageAssignment(None, target_id, 1.0, 0.0, 0.0, 0.0, 0.0)

    pool = _coverage_pool(target, defenders)
    # Primary responsibility follows participation first, then skill. This still lets elite
    # corners matter without pretending they cover every receiver/TE/RB on every snap.
    ranked = sorted(
        pool,
        key=lambda d: (
            max(float(getattr(d, "snap_weight", 1.0)), 0.01),
            float(getattr(d, "coverage", 1.0)),
        ),
        reverse=True,
    )
    primary = ranked[0]

    # Blend the primary assignment with the plausible coverage rotation. The local defender
    # remains explicit, while the mean is structurally centered on who actually plays rather
    # than the maximum Madden rating in the secondary.
    rotation = tuple(ranked[: min(4, len(ranked))])
    rotation_coverage = _weighted_mean(rotation, "coverage")
    primary_coverage = float(getattr(primary, "coverage", 1.0))
    local = float(np.clip(0.62 * primary_coverage + 0.38 * rotation_coverage, 0.65, 1.40))

    target_route = float(getattr(target, "efficiency", 1.0))
    separation = _edge(target_route, local)

    safeties = tuple(
        d
        for d in defenders
        if str(getattr(d, "position", "")).upper() in {"S", "FS", "SS", "DB"}
        and d is not primary
    )
    safety_coverage = _weighted_mean(safeties, "coverage") if safeties else 1.0
    safety_presence = (
        float(np.clip(np.mean([float(getattr(d, "snap_weight", 1.0)) for d in safeties]), 0.0, 1.0))
        if safeties
        else 0.0
    )
    safety_help = float(np.clip((safety_coverage - 0.92) * 0.45 * safety_presence, 0.0, 0.18))

    explosive = float(getattr(target, "explosive", 1.0))
    second = ranked[1] if len(ranked) > 1 else None
    bracket = 0.0
    if second is not None and explosive > 1.02:
        bracket = float(
            np.clip(
                (float(getattr(second, "coverage", 1.0)) - 0.92)
                * 0.28
                * max(float(getattr(second, "snap_weight", 1.0)), 0.05),
                0.0,
                0.12,
            )
        )
    zone_overlap = float(
        np.clip((rotation_coverage - 0.94) * 0.32, 0.0, 0.12)
    )
    return CoverageAssignment(
        str(getattr(primary, "player_id", "")) or None,
        target_id,
        local,
        separation,
        safety_help,
        bracket,
        zone_overlap,
    )


def resolve_pass_snap_v2(
    *, target: object, defense: object, pass_protection: float, quarterback_efficiency: float
) -> PassSnapResolution:
    defenders = tuple(getattr(defense, "coverage", ()))
    coverage = resolve_coverage_assignment_v2(target=target, defenders=defenders)
    pressure, ttp, pocket, help_strength, rusher_id, _ = resolve_pass_protection(
        target_id=str(getattr(target, "player_id", "")),
        rushers=tuple(getattr(defense, "front", ())),
        base_pressure_rate=float(getattr(defense, "pressure_rate", 0.297832)),
        fallback_pass_protection=pass_protection,
    )
    qb_read = resolve_qb_read(
        quarterback_efficiency=quarterback_efficiency,
        separation_edge=coverage.separation_edge,
        time_to_pressure=ttp,
        safety_help=coverage.safety_help,
        bracket_factor=coverage.bracket_factor,
        zone_overlap=coverage.zone_overlap,
    )
    # Help/overlap remain their own causal channels. Do not bury the same safety/bracket
    # difficulty inside coverage strength and then charge it again downstream.
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
    )


def resolve_run_snap_v2(*, rusher: object, defense: object, run_blocking: float) -> RunSnapResolution:
    profile = team_profile_for_player(str(getattr(rusher, "player_id", "")))
    blockers = () if profile is None else profile.offensive_line
    if blockers:
        block_values = np.asarray([float(b.run_block) for b in blockers], dtype=float)
        block_weights = np.asarray([max(float(b.snap_weight), 0.05) for b in blockers], dtype=float)
        lane_blocking = float(np.average(block_values, weights=block_weights))
    else:
        lane_blocking = float(run_blocking)

    front = tuple(getattr(defense, "front", ()))
    coverage = tuple(getattr(defense, "coverage", ()))
    if front:
        # A real run sees the participating front, not the four highest run-defense ratings on
        # the roster every snap. Madden still determines each defender's skill and the primary
        # threat; snap share determines how much each defender contributes to the fit.
        front_fit = _weighted_mean(front, "run_defense")
        primary = max(
            front,
            key=lambda d: float(getattr(d, "run_defense", 1.0))
            * max(float(getattr(d, "snap_weight", 1.0)), 0.05),
        )
    else:
        front_fit, primary = 1.0, None

    second_pool = coverage if coverage else front
    second_level = _weighted_mean(second_pool, "tackling") if second_pool else 1.0
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
        None if primary is None else str(getattr(primary, "player_id", "")) or None,
    )
