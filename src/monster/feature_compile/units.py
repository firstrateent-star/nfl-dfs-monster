from __future__ import annotations

from dataclasses import dataclass, replace
from math import tanh

import numpy as np

from monster.snapshot.model import TeamState


@dataclass(frozen=True)
class UnitPlayerInputs:
    """Simulation-facing evidence for any rostered player, not only fantasy positions."""

    player_id: str
    position: str
    offense_snap_share: float = 0.0
    defense_snap_share: float = 0.0
    special_teams_snap_share: float = 0.0
    snap_share_uncertainty: float = 0.0
    active_probability: float = 1.0
    effectiveness_if_active: float = 1.0

    madden_pass_block: float | None = None
    madden_run_block: float | None = None
    madden_pass_rush: float | None = None
    madden_coverage: float | None = None
    madden_tackle: float | None = None
    madden_speed: float | None = None
    madden_kick_power: float | None = None
    madden_kick_accuracy: float | None = None
    madden_return: float | None = None

    pass_block_signal: float | None = None
    run_block_signal: float | None = None
    pass_rush_signal: float | None = None
    coverage_signal: float | None = None
    run_defense_signal: float | None = None
    special_teams_signal: float | None = None


@dataclass(frozen=True)
class TeamUnitEffects:
    pass_protection_effect: float = 0.0
    run_block_effect: float = 0.0
    pass_rush_effect: float = 0.0
    coverage_effect: float = 0.0
    run_defense_effect: float = 0.0
    special_teams_effect: float = 0.0
    personnel_uncertainty: float = 0.0


@dataclass(frozen=True)
class UnitTrace:
    offense_snap_weight: float
    defense_snap_weight: float
    special_teams_snap_weight: float
    players_with_offense_weight: int
    players_with_defense_weight: int
    players_with_special_teams_weight: int


def _rating_signal(value: float | None, center: float = 78.0, scale: float = 10.0) -> float:
    if value is None:
        return 0.0
    return float(tanh((float(value) - center) / scale))


def _bounded_signal(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _blend(primary: float | None, madden: float | None, *, madden_weight: float = 0.35) -> float:
    observed = _bounded_signal(primary)
    proxy = _rating_signal(madden)
    if primary is None and madden is None:
        return 0.0
    if primary is None:
        return proxy
    if madden is None:
        return observed
    return float(np.clip((1.0 - madden_weight) * observed + madden_weight * proxy, -1.0, 1.0))


def _availability_weight(player: UnitPlayerInputs, snap_share: float) -> float:
    return float(
        np.clip(snap_share, 0.0, 1.0)
        * np.clip(player.active_probability, 0.0, 1.0)
        * np.clip(player.effectiveness_if_active, 0.35, 1.10)
    )


def _weighted_average(values: list[tuple[float, float]]) -> tuple[float, float]:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0.0:
        return 0.0, 0.0
    return float(sum(value * weight for value, weight in values) / total_weight), float(total_weight)


def _position_group(position: str) -> str:
    p = position.upper().strip()
    if p in {"OT", "T", "LT", "RT", "OG", "G", "LG", "RG", "C", "OL"}:
        return "OL"
    if p in {"DE", "DT", "NT", "DL", "EDGE"}:
        return "DL"
    if p in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if p in {"CB", "DB", "S", "FS", "SS"}:
        return "DB"
    if p in {"K", "P", "LS", "SPEC"}:
        return "SPEC"
    return p


def compile_team_unit_effects(
    players: tuple[UnitPlayerInputs, ...],
) -> tuple[TeamUnitEffects, UnitTrace]:
    """Aggregate participating players only into mechanisms their positions can inform."""
    pass_block_values: list[tuple[float, float]] = []
    run_block_values: list[tuple[float, float]] = []
    pass_rush_values: list[tuple[float, float]] = []
    coverage_values: list[tuple[float, float]] = []
    run_defense_values: list[tuple[float, float]] = []
    special_values: list[tuple[float, float]] = []
    uncertainty_values: list[tuple[float, float]] = []

    offense_players = defense_players = special_players = 0
    offense_trace_weight = defense_trace_weight = special_trace_weight = 0.0

    for player in players:
        group = _position_group(player.position)
        offense_weight = _availability_weight(player, player.offense_snap_share)
        defense_weight = _availability_weight(player, player.defense_snap_share)
        special_weight = _availability_weight(player, player.special_teams_snap_share)

        if offense_weight > 0.0:
            offense_players += 1
            offense_trace_weight += offense_weight
            # Protection is dominated by OL, with TE/RB secondary blocking jurisdiction.
            if group in {"OL", "TE", "RB"}:
                jurisdiction = 1.0 if group == "OL" else 0.35
                pass_block_values.append(
                    (
                        _blend(player.pass_block_signal, player.madden_pass_block),
                        offense_weight * jurisdiction,
                    )
                )
            # Run blocking also includes TE/WR edge blocking, but at lower authority.
            if group in {"OL", "TE", "WR"}:
                jurisdiction = 1.0 if group == "OL" else (0.40 if group == "TE" else 0.15)
                run_block_values.append(
                    (
                        _blend(player.run_block_signal, player.madden_run_block),
                        offense_weight * jurisdiction,
                    )
                )
            uncertainty_values.append((player.snap_share_uncertainty, offense_weight))

        if defense_weight > 0.0:
            defense_players += 1
            defense_trace_weight += defense_weight
            if group in {"DL", "LB"}:
                jurisdiction = 1.0 if group == "DL" else 0.55
                pass_rush_values.append(
                    (
                        _blend(player.pass_rush_signal, player.madden_pass_rush),
                        defense_weight * jurisdiction,
                    )
                )
            if group in {"DB", "LB"}:
                jurisdiction = 1.0 if group == "DB" else 0.55
                coverage_proxy = player.madden_coverage
                if coverage_proxy is None and player.madden_speed is not None:
                    coverage_proxy = 0.70 * 78.0 + 0.30 * player.madden_speed
                coverage_values.append(
                    (
                        _blend(player.coverage_signal, coverage_proxy),
                        defense_weight * jurisdiction,
                    )
                )
            if group in {"DL", "LB", "DB"}:
                jurisdiction = 1.0 if group in {"DL", "LB"} else 0.45
                run_defense_values.append(
                    (
                        _blend(player.run_defense_signal, player.madden_tackle),
                        defense_weight * jurisdiction,
                    )
                )
            uncertainty_values.append((player.snap_share_uncertainty, defense_weight))

        if special_weight > 0.0:
            special_players += 1
            special_trace_weight += special_weight
            kick_signal = 0.0
            if player.madden_kick_accuracy is not None or player.madden_kick_power is not None:
                kick_signal = 0.60 * _rating_signal(player.madden_kick_accuracy) + 0.40 * _rating_signal(
                    player.madden_kick_power
                )
            return_signal = _rating_signal(player.madden_return)
            explicit = _bounded_signal(player.special_teams_signal)
            signal = float(
                np.clip(0.55 * explicit + 0.30 * kick_signal + 0.15 * return_signal, -1.0, 1.0)
            )
            special_values.append((signal, special_weight))
            uncertainty_values.append((player.snap_share_uncertainty, special_weight))

    pass_block, _ = _weighted_average(pass_block_values)
    run_block, _ = _weighted_average(run_block_values)
    pass_rush, _ = _weighted_average(pass_rush_values)
    coverage, _ = _weighted_average(coverage_values)
    run_defense, _ = _weighted_average(run_defense_values)
    special, _ = _weighted_average(special_values)
    uncertainty, _ = _weighted_average(uncertainty_values)

    effects = TeamUnitEffects(
        pass_protection_effect=float(np.clip(0.04 * pass_block, -0.04, 0.04)),
        run_block_effect=float(np.clip(0.04 * run_block, -0.04, 0.04)),
        pass_rush_effect=float(np.clip(0.04 * pass_rush, -0.04, 0.04)),
        coverage_effect=float(np.clip(0.04 * coverage, -0.04, 0.04)),
        run_defense_effect=float(np.clip(0.04 * run_defense, -0.04, 0.04)),
        special_teams_effect=float(np.clip(0.025 * special, -0.025, 0.025)),
        personnel_uncertainty=float(np.clip(uncertainty, 0.0, 0.20)),
    )
    trace = UnitTrace(
        offense_snap_weight=offense_trace_weight,
        defense_snap_weight=defense_trace_weight,
        special_teams_snap_weight=special_trace_weight,
        players_with_offense_weight=offense_players,
        players_with_defense_weight=defense_players,
        players_with_special_teams_weight=special_players,
    )
    return effects, trace


def apply_team_unit_effects(team: TeamState, effects: TeamUnitEffects) -> TeamState:
    """Attach snap-weighted all-player unit effects to a team simulation state."""
    return replace(
        team,
        pass_protection_effect=effects.pass_protection_effect,
        run_block_effect=effects.run_block_effect,
        pass_rush_effect=effects.pass_rush_effect,
        coverage_effect=effects.coverage_effect,
        run_defense_effect=effects.run_defense_effect,
        special_teams_effect=effects.special_teams_effect,
        uncertainty=float(np.clip(team.uncertainty + 0.25 * effects.personnel_uncertainty, 0.04, 0.35)),
    )
