from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, tanh

import numpy as np

from monster.sim.play_kernel import PlayerIdentity
from monster.sim.rich_identity import (
    catch_skill,
    open_field_skill,
    route_skill,
    rush_creation_skill,
    skill_multiplier,
)
from monster.sim.snap_ecology_v4b import resolve_pass_snap, resolve_run_snap

# 2025 regular-season FTN participation via nflverse, measured on qb_dropback plays.
# League priors anchor the center; explicit assigned player duels create the matchup spread.
LEAGUE_PRESSURE_RATE = 0.297832
LEAGUE_THROW_COMPLETION_RATE = 0.642661
LEAGUE_THROW_INTERCEPTION_RATE = 0.021772


@dataclass(frozen=True)
class DefensiveIdentity:
    player_id: str
    name: str
    position: str
    coverage: float = 1.0
    pass_rush: float = 1.0
    run_defense: float = 1.0
    tackling: float = 1.0
    ball_hawk: float = 1.0
    speed: float = 1.0
    returning: float = 1.0
    snap_weight: float = 1.0


@dataclass(frozen=True)
class DefensiveUnit:
    front: tuple[DefensiveIdentity, ...]
    coverage: tuple[DefensiveIdentity, ...]
    pressure_rate: float = LEAGUE_PRESSURE_RATE
    run_stuff_rate: float = 0.18


@dataclass(frozen=True)
class PassMatchup:
    pressure_probability: float
    interception_probability: float
    completion_probability: float
    yards_multiplier: float
    primary_defender_id: str | None
    coverage_strength: float = 1.0
    ball_hawk_strength: float = 1.0
    local_coverage_strength: float = 1.0
    local_separation_edge: float = 0.0
    local_rush_strength: float = 1.0
    time_to_pressure: float = 2.55
    pocket_integrity: float = 1.0
    protection_help: float = 0.0
    safety_help: float = 0.0
    bracket_factor: float = 0.0
    zone_overlap: float = 0.0
    qb_read_quality: float = 1.0
    primary_rusher_id: str | None = None
    safety_defender_id: str | None = None
    bracket_defender_id: str | None = None

    @property
    def participant_ids(self) -> frozenset[str]:
        return frozenset(
            player_id
            for player_id in (
                self.primary_defender_id,
                self.primary_rusher_id,
                self.safety_defender_id,
                self.bracket_defender_id,
            )
            if player_id
        )


@dataclass(frozen=True)
class RunMatchup:
    stuff_probability: float
    yards_multiplier: float
    primary_defender_id: str | None
    local_run_defense: float = 1.0
    second_level_tackling: float = 1.0
    runner_edge: float = 0.0
    lane_blocking: float = 1.0
    front_fit: float = 1.0
    pursuit_defender_id: str | None = None

    @property
    def participant_ids(self) -> frozenset[str]:
        return frozenset(
            player_id
            for player_id in (self.primary_defender_id, self.pursuit_defender_id)
            if player_id
        )


def _unit_strength(defenders: tuple[DefensiveIdentity, ...], attribute: str) -> float:
    """Exposure-weighted context around the actual assigned local interaction."""
    if not defenders:
        return 1.0
    weights = np.asarray(
        [max(defender.snap_weight, 0.001) for defender in defenders], dtype=float
    )
    values = np.asarray(
        [float(getattr(defender, attribute)) for defender in defenders], dtype=float
    )
    return float(np.average(values, weights=weights))


def _representative_defender(
    defenders: tuple[DefensiveIdentity, ...],
) -> DefensiveIdentity | None:
    """Fallback participation proxy based on exposure only, never player skill."""
    if not defenders:
        return None
    return max(defenders, key=lambda defender: (defender.snap_weight, defender.player_id))


def _bounded_relative_product(*terms: tuple[float, float], low: float, high: float) -> float:
    """Combine relative football edges without allowing many small effects to collapse a prior."""
    log_relative = 0.0
    for value, authority in terms:
        log_relative += authority * log(max(float(value), 1e-6))
    return float(np.clip(exp(log_relative), low, high))


def resolve_pass_matchup(
    target: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    pass_protection: float,
    quarterback_efficiency: float,
    responsibility_key: str = "static",
) -> PassMatchup:
    """Resolve one receiver against the shared defensive world for the current snap.

    v6 makes the assigned receiver/defender and rusher/blocker duels primary.  Unit quality,
    safety help and zone overlap remain real context, but they no longer average away the
    skill difference between the players actually involved in the play.
    """
    snap = resolve_pass_snap(
        target=target,
        defense=defense,
        pass_protection=pass_protection,
        quarterback_efficiency=quarterback_efficiency,
        responsibility_key=responsibility_key,
    )
    coverage_unit = _unit_strength(defense.coverage, "coverage")
    ball_hawk_unit = _unit_strength(defense.coverage, "ball_hawk")
    rush_unit = _unit_strength(defense.front, "pass_rush")

    cover = next(
        (d for d in defense.coverage if d.player_id == snap.primary_defender_id),
        _representative_defender(defense.coverage),
    )
    local_coverage = 1.0 if cover is None else float(cover.coverage)
    local_ball_hawk = 1.0 if cover is None else float(cover.ball_hawk)
    local_rusher = next(
        (d for d in defense.front if d.player_id == snap.primary_rusher_id),
        _representative_defender(defense.front),
    )
    local_rush = 1.0 if local_rusher is None else float(local_rusher.pass_rush)

    help_context = float(
        np.clip(
            1.0
            + 0.55 * snap.safety_help
            + 0.75 * snap.bracket_factor
            + 0.45 * snap.zone_overlap,
            0.92,
            1.42,
        )
    )
    effective_coverage = _bounded_relative_product(
        (local_coverage, 0.72),
        (coverage_unit, 0.12),
        (help_context, 0.55),
        low=0.48,
        high=1.70,
    )
    effective_ball_hawk = _bounded_relative_product(
        (local_ball_hawk, 0.72),
        (ball_hawk_unit, 0.14),
        (1.0 + snap.zone_overlap, 0.28),
        low=0.50,
        high=1.62,
    )
    effective_rush = _bounded_relative_product(
        (local_rush, 0.72),
        (rush_unit, 0.12),
        low=0.50,
        high=1.68,
    )

    pressure = float(
        np.clip(
            0.86 * snap.pressure_probability
            + 0.14
            * defense.pressure_rate
            * effective_rush
            / max(snap.pocket_integrity, 0.58),
            0.05,
            0.68,
        )
    )

    speed_signal = float(np.clip(getattr(target, "speed_skill", 0.0), -1.0, 1.0))
    target_catch = catch_skill(target)
    target_route = route_skill(target)
    target_open_field = open_field_skill(target)

    # Direct route-runner vs assigned coverage defender duel.  The edge is intentionally
    # bounded and neutral-centered; it changes mechanism probabilities instead of adding yards.
    route_vs_cover = float(tanh((target_route - local_coverage) / 0.18))
    identity_separation = float(
        np.clip(0.82 * route_vs_cover + 0.18 * speed_signal, -1.0, 1.0)
    )
    separation_factor = float(np.clip(exp(0.18 * identity_separation), 0.82, 1.22))

    completion_relative = _bounded_relative_product(
        (np.clip(snap.qb_read_quality, 0.62, 1.45), 0.46),
        (np.clip(target_catch, 0.60, 1.48), 0.42),
        (np.clip(target_route, 0.60, 1.48), 0.14),
        (1.0 / max(effective_coverage, 0.52), 0.62),
        (separation_factor, 1.0),
        low=0.62,
        high=1.46,
    )
    completion = float(
        np.clip(
            LEAGUE_THROW_COMPLETION_RATE * completion_relative,
            0.20,
            0.92,
        )
    )

    interception_relative = _bounded_relative_product(
        (effective_ball_hawk, 0.62),
        (effective_coverage, 0.30),
        (1.0 / max(snap.qb_read_quality, 0.52), 0.42),
        (float(np.clip(exp(-0.16 * identity_separation), 0.84, 1.18)), 1.0),
        low=0.45,
        high=2.05,
    )
    interception = float(
        np.clip(
            LEAGUE_THROW_INTERCEPTION_RATE * interception_relative,
            0.003,
            0.085,
        )
    )

    yards_multiplier = float(
        np.clip(
            target_open_field
            / max(effective_coverage**0.28, 0.68)
            * exp(0.22 * identity_separation)
            * (1.0 - 0.10 * snap.safety_help - 0.16 * snap.bracket_factor),
            0.44,
            1.95,
        )
    )
    return PassMatchup(
        pressure_probability=pressure,
        interception_probability=interception,
        completion_probability=completion,
        yards_multiplier=yards_multiplier,
        primary_defender_id=snap.primary_defender_id,
        coverage_strength=effective_coverage,
        ball_hawk_strength=effective_ball_hawk,
        local_coverage_strength=local_coverage,
        local_separation_edge=identity_separation,
        local_rush_strength=local_rush,
        time_to_pressure=snap.time_to_pressure,
        pocket_integrity=snap.pocket_integrity,
        protection_help=snap.protection_help,
        safety_help=snap.safety_help,
        bracket_factor=snap.bracket_factor,
        zone_overlap=snap.zone_overlap,
        qb_read_quality=snap.qb_read_quality,
        primary_rusher_id=snap.primary_rusher_id,
        safety_defender_id=snap.safety_defender_id,
        bracket_defender_id=snap.bracket_defender_id,
    )


def resolve_run_matchup(
    rusher: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    run_blocking: float,
    responsibility_key: str = "static",
    run_geometry: str | None = None,
) -> RunMatchup:
    """Resolve geometry-specific blockers, front fit and second-level pursuit."""
    snap = resolve_run_snap(
        rusher=rusher,
        defense=defense,
        run_blocking=run_blocking,
        responsibility_key=responsibility_key,
        run_geometry=run_geometry,
    )
    primary = next(
        (d for d in defense.front if d.player_id == snap.primary_defender_id),
        _representative_defender(defense.front),
    )
    local_run_defense = 1.0 if primary is None else float(primary.run_defense)

    creation = rush_creation_skill(rusher)
    open_field = open_field_skill(rusher)
    power = skill_multiplier(rusher, "runner_power", 0.20)
    runner_signal = float(np.clip(getattr(rusher, "rush_creation_skill", 0.0), -1.0, 1.0))
    runner_edge = float(np.clip(snap.runner_edge + 0.38 * runner_signal, -1.0, 1.0))
    stuff = float(
        np.clip(
            snap.stuff_probability / max(power**0.48, 0.84),
            0.025,
            0.56,
        )
    )
    yards = float(
        np.clip(
            snap.yards_multiplier
            * (
                creation
                / max(float(getattr(rusher, "efficiency", 1.0)), 0.55)
            )
            ** 0.62
            * (
                open_field
                / max(float(getattr(rusher, "explosive", 1.0)), 0.55)
            )
            ** 0.30,
            0.38,
            2.05,
        )
    )
    return RunMatchup(
        stuff_probability=stuff,
        yards_multiplier=yards,
        primary_defender_id=snap.primary_defender_id,
        local_run_defense=local_run_defense,
        second_level_tackling=snap.second_level_fit,
        runner_edge=runner_edge,
        lane_blocking=snap.lane_blocking,
        front_fit=snap.front_fit,
        pursuit_defender_id=snap.pursuit_defender_id,
    )
