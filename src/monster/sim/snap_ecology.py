from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np


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


@dataclass(frozen=True)
class RunSnapResolution:
    lane_blocking: float
    front_fit: float
    second_level_fit: float
    runner_edge: float
    stuff_probability: float
    yards_multiplier: float
    primary_defender_id: str | None


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


def _signal(value: object, *, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    if -1.5 <= x <= 1.5:
        return float(np.clip(1.0 + 0.18 * x, 0.72, 1.28))
    return _rating(x)


def _position_order(position: str) -> int:
    order = {"LT": 0, "LG": 1, "C": 2, "RG": 3, "RT": 4, "T": 5, "G": 6, "OL": 7}
    return order.get(position.upper(), 99)


def register_team_units(team_id: str, players: Iterable[object]) -> TeamSnapProfile:
    """Register current player-level blocking/protection evidence for snap resolution.

    The registry is football-only state. It is populated from the frozen personnel snapshot
    before worlds are simulated and contains no salary, ownership or sportsbook information.
    """
    rows = list(players)
    for row in rows:
        player_id = str(getattr(row, "player_id", ""))
        if player_id:
            _PLAYER_TEAM[player_id] = team_id

    ol_rows = [
        row
        for row in rows
        if str(getattr(row, "position", "")).upper() in {"LT", "LG", "C", "RG", "RT", "T", "G", "OT", "OG", "OL"}
    ]
    ol_rows.sort(
        key=lambda row: (
            _position_order(str(getattr(row, "position", ""))),
            -float(getattr(row, "offense_snap_share", 0.0) or 0.0),
        )
    )
    selected = ol_rows[:5]
    blockers = tuple(
        BlockerProfile(
            player_id=str(getattr(row, "player_id", "")),
            position=str(getattr(row, "position", "OL")),
            pass_block=_signal(
                getattr(row, "pass_block_signal", None),
                default=_rating(getattr(row, "madden_pass_block", None)),
            ),
            run_block=_signal(
                getattr(row, "run_block_signal", None),
                default=_rating(getattr(row, "madden_run_block", None)),
            ),
            awareness=_rating(getattr(row, "madden_awareness", None)),
            stamina=_rating(getattr(row, "madden_stamina", None), center=82.0, scale=10.0),
            snap_weight=float(np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)),
        )
        for row in selected
    )

    help_rows = [
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
                np.clip(
                    np.mean(
                        [
                            _rating(getattr(row, "madden_route_running", None)),
                            _rating(getattr(row, "madden_catching", None)),
                        ]
                    ),
                    0.72,
                    1.28,
                )
            ),
            snap_weight=float(np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)),
        )
        for row in help_rows
    )
    profile = TeamSnapProfile(team_id=team_id, offensive_line=blockers, protectors=protectors)
    _TEAM_PROFILES[team_id] = profile
    return profile


def team_profile_for_player(player_id: str) -> TeamSnapProfile | None:
    team = _PLAYER_TEAM.get(player_id)
    return None if team is None else _TEAM_PROFILES.get(team)


def _duel_edge(offense: float, defense: float, scale: float = 0.22) -> float:
    return float(np.tanh((offense - defense) / max(scale, 1e-6)))


def _best_help(profile: TeamSnapProfile | None) -> ProtectorProfile | None:
    if profile is None or not profile.protectors:
        return None
    return max(profile.protectors, key=lambda p: p.pass_block * max(p.snap_weight, 0.05))


def resolve_pass_protection(
    *,
    target_id: str,
    rushers: tuple[object, ...],
    base_pressure_rate: float,
    fallback_pass_protection: float,
) -> tuple[float, float, float, float, str | None, tuple[PassRushDuel, ...]]:
    """Resolve five blocker-v-rusher duels plus optional RB/TE protection help."""
    profile = team_profile_for_player(target_id)
    if profile is None or not profile.offensive_line or not rushers:
        pressure = float(np.clip(base_pressure_rate / max(fallback_pass_protection, 0.55), 0.08, 0.58))
        ttp = float(np.clip(2.65 - 1.9 * (pressure - 0.20), 1.35, 3.25))
        return pressure, ttp, fallback_pass_protection, 0.0, None, ()

    blockers = profile.offensive_line
    ordered_rushers = sorted(
        rushers,
        key=lambda d: float(getattr(d, "pass_rush", 1.0)) * max(float(getattr(d, "snap_weight", 1.0)), 0.05),
        reverse=True,
    )
    duels: list[PassRushDuel] = []
    for i, rusher in enumerate(ordered_rushers[: min(len(ordered_rushers), 5)]):
        blocker = blockers[min(i, len(blockers) - 1)]
        rusher_strength = float(getattr(rusher, "pass_rush", 1.0))
        blocker_strength = float(np.clip(0.70 * blocker.pass_block + 0.20 * blocker.awareness + 0.10 * blocker.stamina, 0.65, 1.35))
        edge = _duel_edge(rusher_strength, blocker_strength)
        ttp = float(np.clip(2.75 - 0.72 * edge, 1.25, 3.55))
        duels.append(
            PassRushDuel(
                blocker_id=blocker.player_id,
                rusher_id=str(getattr(rusher, "player_id", "")) or None,
                blocker_strength=blocker_strength,
                rusher_strength=rusher_strength,
                rush_edge=edge,
                time_to_pressure=ttp,
            )
        )

    help_player = _best_help(profile)
    help_strength = 0.0 if help_player is None else max(help_player.pass_block - 0.80, 0.0)
    if duels:
        most_dangerous = max(range(len(duels)), key=lambda i: duels[i].rush_edge)
        duel = duels[most_dangerous]
        helped_edge = float(np.clip(duel.rush_edge - 0.32 * help_strength, -1.0, 1.0))
        duels[most_dangerous] = PassRushDuel(
            blocker_id=duel.blocker_id,
            rusher_id=duel.rusher_id,
            blocker_strength=duel.blocker_strength,
            rusher_strength=duel.rusher_strength,
            rush_edge=helped_edge,
            time_to_pressure=float(np.clip(duel.time_to_pressure + 0.30 * help_strength, 1.25, 3.70)),
        )

    worst_edge = max((d.rush_edge for d in duels), default=0.0)
    mean_edge = float(np.mean([d.rush_edge for d in duels])) if duels else 0.0
    pocket = float(np.clip(1.0 - 0.20 * worst_edge - 0.09 * mean_edge + 0.06 * help_strength, 0.68, 1.30))
    pressure = float(
        np.clip(
            base_pressure_rate
            * (1.0 + 0.34 * worst_edge + 0.18 * mean_edge)
            / max(pocket, 0.65),
            0.07,
            0.62,
        )
    )
    time_to_pressure = float(min((d.time_to_pressure for d in duels), default=2.55))
    primary = None
    if duels:
        primary = min(duels, key=lambda d: d.time_to_pressure).rusher_id
    return pressure, time_to_pressure, pocket, help_strength, primary, tuple(duels)


def resolve_coverage_assignment(
    *,
    target: object,
    defenders: tuple[object, ...],
) -> CoverageAssignment:
    """Resolve target assignment, safety help, bracket pressure and zone overlap."""
    target_id = str(getattr(target, "player_id", ""))
    target_skill = float(getattr(target, "efficiency", 1.0))
    target_explosive = float(getattr(target, "explosive", 1.0))
    if not defenders:
        return CoverageAssignment(None, target_id, 1.0, 0.0, 0.0, 0.0, 0.0)

    ranked = sorted(
        defenders,
        key=lambda d: float(getattr(d, "coverage", 1.0)) * max(float(getattr(d, "snap_weight", 1.0)), 0.05),
        reverse=True,
    )
    primary = ranked[0]
    local = float(getattr(primary, "coverage", 1.0))
    edge = _duel_edge(target_skill, local)

    safeties = [d for d in ranked[1:] if str(getattr(d, "position", "")).upper() in {"S", "FS", "SS", "DB"}]
    safety_help = 0.0
    if safeties:
        safety_help = float(np.clip(max(float(getattr(d, "coverage", 1.0)) for d in safeties) - 0.82, 0.0, 0.40))
    second = ranked[1] if len(ranked) > 1 else None
    bracket = 0.0
    if second is not None and target_explosive > 1.02:
        bracket = float(np.clip((float(getattr(second, "coverage", 1.0)) - 0.85) * 0.45, 0.0, 0.22))
    zone_overlap = float(
        np.clip(
            np.mean([float(getattr(d, "coverage", 1.0)) for d in ranked[:4]]) - 0.92,
            0.0,
            0.22,
        )
    )
    return CoverageAssignment(
        defender_id=str(getattr(primary, "player_id", "")) or None,
        target_id=target_id,
        local_coverage=local,
        separation_edge=edge,
        safety_help=safety_help,
        bracket_factor=bracket,
        zone_overlap=zone_overlap,
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
    """Translate read clarity and available pocket time into bounded QB execution."""
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
    coverage = resolve_coverage_assignment(target=target, defenders=tuple(getattr(defense, "coverage", ())))
    pressure, ttp, pocket, protection_help, primary_rusher, _ = resolve_pass_protection(
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
    coverage_strength = float(
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
        pressure_probability=pressure,
        time_to_pressure=ttp,
        pocket_integrity=pocket,
        protection_help=protection_help,
        coverage_strength=coverage_strength,
        separation_edge=coverage.separation_edge,
        safety_help=coverage.safety_help,
        bracket_factor=coverage.bracket_factor,
        zone_overlap=coverage.zone_overlap,
        primary_defender_id=coverage.defender_id,
        primary_rusher_id=primary_rusher,
        qb_read_quality=qb_read,
    )


def resolve_run_snap(
    *,
    rusher: object,
    defense: object,
    run_blocking: float,
) -> RunSnapResolution:
    """Resolve five-man lane blocking against front fit and second-level pursuit."""
    player_id = str(getattr(rusher, "player_id", ""))
    profile = team_profile_for_player(player_id)
    blockers = () if profile is None else profile.offensive_line
    blocker_strengths = [b.run_block for b in blockers]
    lane_blocking = float(np.mean(blocker_strengths)) if blocker_strengths else run_blocking

    front = tuple(getattr(defense, "front", ()))
    coverage = tuple(getattr(defense, "coverage", ()))
    if front:
        front_ranked = sorted(front, key=lambda d: float(getattr(d, "run_defense", 1.0)), reverse=True)
        front_fit = float(np.mean([float(getattr(d, "run_defense", 1.0)) for d in front_ranked[:4]]))
        primary = front_ranked[0]
    else:
        front_fit = 1.0
        primary = None
    second_pool = coverage if coverage else front
    second_level = (
        float(np.mean([float(getattr(d, "tackling", 1.0)) for d in second_pool[:4]]))
        if second_pool
        else 1.0
    )
    runner_skill = float(getattr(rusher, "efficiency", 1.0))
    runner_edge = _duel_edge(runner_skill * lane_blocking, front_fit)
    base_stuff = float(getattr(defense, "run_stuff_rate", 0.18))
    stuff = float(
        np.clip(
            base_stuff
            * front_fit
            / max(lane_blocking, 0.55)
            * (1.0 - 0.14 * runner_edge),
            0.04,
            0.50,
        )
    )
    yards_multiplier = float(
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
        lane_blocking=lane_blocking,
        front_fit=front_fit,
        second_level_fit=second_level,
        runner_edge=runner_edge,
        stuff_probability=stuff,
        yards_multiplier=yards_multiplier,
        primary_defender_id=None if primary is None else str(getattr(primary, "player_id", "")) or None,
    )
