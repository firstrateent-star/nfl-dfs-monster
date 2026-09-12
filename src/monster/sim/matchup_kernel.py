from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

from monster.sim.play_kernel import PlayerIdentity
from monster.sim.snap_ecology import resolve_pass_snap, resolve_run_snap

# 2025 regular-season FTN participation via nflverse, measured on qb_dropback plays.
# Team/player matchup traits perturb these baselines rather than replacing the causal priors.
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


def _unit_strength(defenders: tuple[DefensiveIdentity, ...], attribute: str) -> float:
    """Exposure-weighted 11-man context; no single defender represents the whole unit."""
    if not defenders:
        return 1.0
    weights = np.asarray([max(defender.snap_weight, 0.001) for defender in defenders], dtype=float)
    values = np.asarray([float(getattr(defender, attribute)) for defender in defenders], dtype=float)
    return float(np.average(values, weights=weights))


def _representative_defender(
    defenders: tuple[DefensiveIdentity, ...], attribute: str
) -> DefensiveIdentity | None:
    if not defenders:
        return None
    return max(
        defenders,
        key=lambda defender: max(defender.snap_weight, 0.001)
        * max(float(getattr(defender, attribute)), 0.001),
    )


def _bounded_relative_product(*terms: tuple[float, float], low: float, high: float) -> float:
    """Combine relative football edges without allowing many small effects to collapse a prior.

    Each term is ``(relative_value, authority)``. Log-space authority means neutral values
    remain exactly neutral, while player/QB/coverage differences still move the outcome in
    the correct direction. This is intentionally different from multiplying several raw
    factors at full authority, which systematically depressed the league completion ecology.
    """
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
) -> PassMatchup:
    """Resolve the target inside a complete protection/coverage snap ecology.

    Five current offensive linemen are registered from the frozen personnel snapshot and
    paired against the most relevant rushers. RB/TE protection can reinforce the most dangerous
    rush lane. Coverage assigns a primary defender while preserving safety help, bracket risk,
    and zone overlap. QB read quality then depends on separation plus the time the pocket buys.

    Throw completion and interception start from empirical 2025 league throw baselines.
    Player/read/coverage evidence moves those priors with bounded relative authority. Pressure
    is *not* charged here because the play kernel subsequently resolves the actual pressure
    state and applies the observed clean-vs-pressure split exactly once.
    """
    snap = resolve_pass_snap(
        target=target,
        defense=defense,
        pass_protection=pass_protection,
        quarterback_efficiency=quarterback_efficiency,
    )
    coverage_unit = _unit_strength(defense.coverage, "coverage")
    ball_hawk_unit = _unit_strength(defense.coverage, "ball_hawk")
    rush_unit = _unit_strength(defense.front, "pass_rush")

    cover = next(
        (d for d in defense.coverage if d.player_id == snap.primary_defender_id),
        _representative_defender(defense.coverage, "coverage"),
    )
    local_coverage = 1.0 if cover is None else float(cover.coverage)
    local_ball_hawk = 1.0 if cover is None else float(cover.ball_hawk)
    local_rusher = next(
        (d for d in defense.front if d.player_id == snap.primary_rusher_id),
        _representative_defender(defense.front, "pass_rush"),
    )
    local_rush = 1.0 if local_rusher is None else float(local_rusher.pass_rush)

    effective_coverage = float(
        np.clip(
            0.42 * coverage_unit
            + 0.34 * snap.coverage_strength
            + 0.24 * local_coverage,
            0.55,
            1.55,
        )
    )
    effective_ball_hawk = float(
        np.clip(
            0.60 * ball_hawk_unit + 0.40 * local_ball_hawk + 0.12 * snap.zone_overlap,
            0.55,
            1.50,
        )
    )
    effective_rush = float(np.clip(0.52 * rush_unit + 0.48 * local_rush, 0.55, 1.55))

    pressure = float(
        np.clip(
            0.70 * snap.pressure_probability
            + 0.30 * defense.pressure_rate * effective_rush / max(snap.pocket_integrity, 0.60),
            0.07,
            0.62,
        )
    )

    separation_factor = float(np.clip(1.0 + 0.06 * snap.separation_edge, 0.88, 1.12))
    completion_relative = _bounded_relative_product(
        (np.clip(snap.qb_read_quality, 0.65, 1.40), 0.42),
        (np.clip(target.efficiency, 0.65, 1.40), 0.32),
        (1.0 / max(effective_coverage, 0.60), 0.46),
        (separation_factor, 1.0),
        low=0.72,
        high=1.30,
    )
    completion = float(
        np.clip(
            LEAGUE_THROW_COMPLETION_RATE * completion_relative,
            0.30,
            0.88,
        )
    )

    interception_relative = _bounded_relative_product(
        (effective_ball_hawk, 0.45),
        (effective_coverage, 0.24),
        (1.0 / max(snap.qb_read_quality, 0.55), 0.34),
        (float(np.clip(1.0 - 0.08 * snap.separation_edge, 0.84, 1.16)), 1.0),
        low=0.55,
        high=1.75,
    )
    interception = float(
        np.clip(
            LEAGUE_THROW_INTERCEPTION_RATE * interception_relative,
            0.004,
            0.075,
        )
    )

    yards_multiplier = float(
        np.clip(
            target.explosive
            / max(effective_coverage**0.34, 0.72)
            * (1.0 + 0.11 * snap.separation_edge)
            * (1.0 - 0.10 * snap.safety_help - 0.14 * snap.bracket_factor),
            0.55,
            1.72,
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
        local_separation_edge=snap.separation_edge,
        local_rush_strength=local_rush,
        time_to_pressure=snap.time_to_pressure,
        pocket_integrity=snap.pocket_integrity,
        protection_help=snap.protection_help,
        safety_help=snap.safety_help,
        bracket_factor=snap.bracket_factor,
        zone_overlap=snap.zone_overlap,
        qb_read_quality=snap.qb_read_quality,
        primary_rusher_id=snap.primary_rusher_id,
    )


def resolve_run_matchup(
    rusher: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    run_blocking: float,
) -> RunMatchup:
    """Resolve runner, five-man blocking lane, front fit and second-level tackling."""
    snap = resolve_run_snap(
        rusher=rusher,
        defense=defense,
        run_blocking=run_blocking,
    )
    primary = next(
        (d for d in defense.front if d.player_id == snap.primary_defender_id),
        _representative_defender(defense.front, "run_defense"),
    )
    local_run_defense = 1.0 if primary is None else float(primary.run_defense)
    return RunMatchup(
        stuff_probability=snap.stuff_probability,
        yards_multiplier=snap.yards_multiplier,
        primary_defender_id=snap.primary_defender_id,
        local_run_defense=local_run_defense,
        second_level_tackling=snap.second_level_fit,
        runner_edge=snap.runner_edge,
        lane_blocking=snap.lane_blocking,
        front_fit=snap.front_fit,
    )