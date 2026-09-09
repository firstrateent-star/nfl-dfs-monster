from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.play_kernel import PlayerIdentity

# 2025 regular-season FTN participation via nflverse, measured on qb_dropback plays.
# Team/player matchup traits perturb this baseline rather than replacing the causal prior.
LEAGUE_PRESSURE_RATE = 0.297832


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


@dataclass(frozen=True)
class RunMatchup:
    stuff_probability: float
    yards_multiplier: float
    primary_defender_id: str | None


def _unit_strength(
    defenders: tuple[DefensiveIdentity, ...], attribute: str
) -> float:
    """Return exposure-weighted unit strength instead of letting one star represent a unit."""
    if not defenders:
        return 1.0
    weights = np.asarray([max(defender.snap_weight, 0.001) for defender in defenders], dtype=float)
    values = np.asarray([float(getattr(defender, attribute)) for defender in defenders], dtype=float)
    return float(np.average(values, weights=weights))


def _representative_defender(
    defenders: tuple[DefensiveIdentity, ...], attribute: str
) -> DefensiveIdentity | None:
    """Choose a stable attribution representative without granting them full unit strength."""
    if not defenders:
        return None
    return max(
        defenders,
        key=lambda defender: max(defender.snap_weight, 0.001)
        * max(float(getattr(defender, attribute)), 0.001),
    )


def resolve_pass_matchup(
    target: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    pass_protection: float,
    quarterback_efficiency: float,
) -> PassMatchup:
    cover = _representative_defender(defense.coverage, "coverage")
    coverage_strength = _unit_strength(defense.coverage, "coverage")
    ball_hawk = _unit_strength(defense.coverage, "ball_hawk")
    rush_strength = _unit_strength(defense.front, "pass_rush")
    pressure = float(
        np.clip(defense.pressure_rate * rush_strength / max(pass_protection, 0.55), 0.12, 0.50)
    )
    completion = float(
        np.clip(
            0.64 * quarterback_efficiency * target.efficiency / max(coverage_strength, 0.60),
            0.34,
            0.84,
        )
    )
    interception = float(
        np.clip(0.022 * ball_hawk * coverage_strength / max(quarterback_efficiency, 0.55), 0.006, 0.07)
    )
    yards_multiplier = float(np.clip(target.explosive / max(coverage_strength**0.35, 0.75), 0.65, 1.55))
    return PassMatchup(
        pressure_probability=pressure,
        interception_probability=interception,
        completion_probability=completion,
        yards_multiplier=yards_multiplier,
        primary_defender_id=None if cover is None else cover.player_id,
        coverage_strength=coverage_strength,
        ball_hawk_strength=ball_hawk,
    )


def resolve_run_matchup(
    rusher: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    run_blocking: float,
) -> RunMatchup:
    defender = _representative_defender(defense.front, "run_defense")
    front_strength = _unit_strength(defense.front, "run_defense")
    stuff = float(
        np.clip(defense.run_stuff_rate * front_strength / max(run_blocking, 0.55), 0.06, 0.42)
    )
    yards_multiplier = float(
        np.clip(rusher.efficiency * run_blocking / max(front_strength, 0.60), 0.55, 1.55)
    )
    return RunMatchup(
        stuff_probability=stuff,
        yards_multiplier=yards_multiplier,
        primary_defender_id=None if defender is None else defender.player_id,
    )