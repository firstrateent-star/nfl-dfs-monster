from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from monster.sim.interaction_topology_v4 import (
    choose_coverage_participants,
    choose_pass_rushers,
    choose_protection_helper,
    choose_run_participants,
    exposure_weighted_mean,
    pair_pass_rushers_to_blockers,
)


@dataclass(frozen=True)
class BlockerProfile:
    player_id: str
    position: str
    pass_block: float = 1.0
    run_block: float = 1.0
    awareness: float = 1.0
    stamina: float = 1.0
    snap_weight: float = 1.0


@dataclass(frozen=True)
class ProtectorProfile:
    player_id: str
    position: str
    pass_block: float = 1.0
    run_block: float = 1.0
    receiving_value: float = 1.0
    snap_weight: float = 1.0


@dataclass(frozen=True)
class TeamSnapProfile:
    team_id: str
    offensive_line: tuple[BlockerProfile, ...]
    protectors: tuple[ProtectorProfile, ...]


@dataclass(frozen=True)
class PassRushDuel:
    blocker_id: str | None
    rusher_id: str | None
    blocker_strength: float
    rusher_strength: float
    rush_edge: float
    time_to_pressure: float


@dataclass(frozen=True)
class CoverageAssignment:
    defender_id: str | None
    target_id: str
    local_coverage: float
    separation_edge: float
    safety_help: float
    bracket_factor: float
    zone_overlap: float
    safety_defender_id: str | None = None
    bracket_defender_id: str | None = None


@dataclass(frozen=True)
class PassSnapResolution:
    pressure_probability: float
    time_to_pressure: float
    pocket_integrity: float
    protection_help: float
    coverage_strength: float
    separation_edge: float
    safety_help: float
    bracket_factor: float
    zone_overlap: float
    primary_defender_id: str | None
    primary_rusher_id: str | None
    qb_read_quality: float
    safety_defender_id: str | None = None
    bracket_defender_id: str | None = None


@dataclass(frozen=True)
class RunSnapResolution:
    lane_blocking: float
    front_fit: float
    second_level_fit: float
    runner_edge: float
    stuff_probability: float
    yards_multiplier: float
    primary_defender_id: str | None
    pursuit_defender_id: str | None = None


_TEAM_PROFILES: dict[str, TeamSnapProfile] = {}
_PLAYER_TEAM: dict[str, str] = {}


def _rating(value: object, center: float = 78.0, scale: float = 10.0) -> float:
    if value is None:
        return 1.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(x):
        return 1.0
    return float(np.clip(1.0 + 0.22 * np.tanh((x - center) / scale), 0.72, 1.28))


def _signal(value: object, fallback: object = None) -> float:
    if value is None:
        return _rating(fallback)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return _rating(fallback)
    if -1.5 <= x <= 1.5:
        return float(np.clip(1.0 + 0.18 * x, 0.72, 1.28))
    return _rating(x)


def _position_order(position: str) -> int:
    return {
        "LT": 0,
        "LG": 1,
        "C": 2,
        "RG": 3,
        "RT": 4,
        "T": 5,
        "OT": 5,
        "G": 6,
        "OG": 6,
        "OL": 7,
    }.get(position.upper(), 99)


def register_team_units(team_id: str, players: Iterable[object]) -> TeamSnapProfile:
    """Freeze current 5-OL and RB/TE protection evidence for snap physics."""
    rows = list(players)
    for row in rows:
        player_id = str(getattr(row, "player_id", ""))
        if player_id:
            _PLAYER_TEAM[player_id] = team_id

    ol = [
        row
        for row in rows
        if str(getattr(row, "position", "")).upper()
        in {"LT", "LG", "C", "RG", "RT", "T", "OT", "G", "OG", "OL"}
    ]
    ol.sort(
        key=lambda row: (
            _position_order(str(getattr(row, "position", ""))),
            -float(getattr(row, "offense_snap_share", 0.0) or 0.0),
        )
    )
    blockers = tuple(
        BlockerProfile(
            player_id=str(getattr(row, "player_id", "")),
            position=str(getattr(row, "position", "OL")),
            pass_block=_signal(
                getattr(row, "pass_block_signal", None),
                getattr(row, "madden_pass_block", None),
            ),
            run_block=_signal(
                getattr(row, "run_block_signal", None),
                getattr(row, "madden_run_block", None),
            ),
            awareness=_rating(getattr(row, "madden_awareness", None)),
            stamina=_rating(getattr(row, "madden_stamina", None), 82.0, 10.0),
            snap_weight=float(
                np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)
            ),
        )
        for row in ol[:5]
    )

    helpers = [
        row
        for row in rows
        if str(getattr(row, "position", "")).upper() in {"RB", "FB", "TE"}
        and float(getattr(row, "offense_snap_share", 0.0) or 0.0) > 0.01
    ]
    protectors = tuple(
        ProtectorProfile(
            player_id=str(getattr(row, "player_id", "")),
            position=str(getattr(row, "position", "")),
            pass_block=_rating(getattr(row, "madden_pass_block", None)),
            run_block=_rating(getattr(row, "madden_run_block", None)),
            receiving_value=float(
                np.mean(
                    [
                        _rating(getattr(row, "madden_route_running", None)),
                        _rating(getattr(row, "madden_catching", None)),
                    ]
                )
            ),
            snap_weight=float(
                np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)
            ),
        )
        for row in helpers
    )
    profile = TeamSnapProfile(team_id, blockers, protectors)
    _TEAM_PROFILES[team_id] = profile
    return profile


def team_profile_for_player(player_id: str) -> TeamSnapProfile | None:
    team = _PLAYER_TEAM.get(player_id)
    return None if team is None else _TEAM_PROFILES.get(team)


def _edge(offense: float, defense: float, scale: float = 0.22) -> float:
    return float(np.tanh((offense - defense) / max(scale, 1e-6)))


def resolve_pass_protection(
    *,
    target_id: str,
    rushers: tuple[object, ...],
    base_pressure_rate: float,
    fallback_pass_protection: float,
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

    rush = choose_pass_rushers(rushers, count=5)
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
        responsibility_key=profile.team_id,
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

    worst = max(d.rush_edge for d in duels)
    mean = float(np.mean([d.rush_edge for d in duels]))
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
    fastest = min(duels, key=lambda d: d.time_to_pressure)
    return (
        pressure,
        fastest.time_to_pressure,
        pocket,
        help_strength,
        fastest.rusher_id,
        tuple(duels),
    )


def _defender_by_id(defenders: tuple[object, ...], player_id: str | None) -> object | None:
    if player_id is None:
        return None
    return next(
        (
            defender
            for defender in defenders
            if str(getattr(defender, "player_id", "")) == player_id
        ),
        None,
    )


def resolve_coverage_assignment(
    *,
    target: object,
    defenders: tuple[object, ...],
) -> CoverageAssignment:
    target_id = str(getattr(target, "player_id", ""))
    if not defenders:
        return CoverageAssignment(None, target_id, 1.0, 0.0, 0.0, 0.0, 0.0)

    participants = choose_coverage_participants(target=target, defenders=defenders)
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


def resolve_qb_read(
    *,
    quarterback_efficiency: float,
    separation_edge: float,
    time_to_pressure: float,
    safety_help: float,
    bracket_factor: float,
    zone_overlap: float,
) -> float:
    time_signal = float(np.tanh((time_to_pressure - 2.45) / 0.55))
    coverage_penalty = 0.32 * safety_help + 0.42 * bracket_factor + 0.30 * zone_overlap
    return float(
        np.clip(
            quarterback_efficiency
            * (1.0 + 0.08 * separation_edge + 0.06 * time_signal - coverage_penalty),
            0.55,
            1.45,
        )
    )


def resolve_pass_snap(
    *,
    target: object,
    defense: object,
    pass_protection: float,
    quarterback_efficiency: float,
) -> PassSnapResolution:
    coverage = resolve_coverage_assignment(
        target=target,
        defenders=tuple(getattr(defense, "coverage", ())),
    )
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
) -> RunSnapResolution:
    profile = team_profile_for_player(str(getattr(rusher, "player_id", "")))
    blockers = () if profile is None else profile.offensive_line
    if blockers:
        weights = np.asarray([max(blocker.snap_weight, 0.001) for blocker in blockers], dtype=float)
        lane_blocking = float(
            np.average(
                np.asarray([blocker.run_block for blocker in blockers], dtype=float),
                weights=weights,
            )
        )
    else:
        lane_blocking = run_blocking

    front = tuple(getattr(defense, "front", ()))
    coverage = tuple(getattr(defense, "coverage", ()))
    participants = choose_run_participants(rusher=rusher, front=front, coverage=coverage)
    primary = _defender_by_id(front, participants.box_defender_id)
    pursuit_pool = coverage if coverage else front
    pursuit = _defender_by_id(pursuit_pool, participants.pursuit_defender_id)

    unit_front_fit = exposure_weighted_mean(front, "run_defense") if front else 1.0
    local_front_fit = (
        1.0 if primary is None else float(getattr(primary, "run_defense", 1.0))
    )
    front_fit = float(np.clip(0.72 * unit_front_fit + 0.28 * local_front_fit, 0.55, 1.55))

    unit_second_level = (
        exposure_weighted_mean(pursuit_pool, "tackling") if pursuit_pool else 1.0
    )
    local_second_level = (
        1.0 if pursuit is None else float(getattr(pursuit, "tackling", 1.0))
    )
    second_level = float(
        np.clip(0.58 * unit_second_level + 0.42 * local_second_level, 0.55, 1.55)
    )

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
