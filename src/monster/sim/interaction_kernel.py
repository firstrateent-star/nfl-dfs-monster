from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity


@dataclass(frozen=True)
class PassInteraction:
    coverage_defender_id: str | None
    pressure_defender_id: str | None
    separation: float
    pressure_probability: float
    completion_probability: float
    interception_probability: float
    yards_multiplier: float


@dataclass(frozen=True)
class RunInteraction:
    box_defender_id: str | None
    pursuit_defender_id: str | None
    penetration_probability: float
    tackle_strength: float
    yards_multiplier: float


def _weighted_choice(
    defenders: tuple[DefensiveIdentity, ...],
    *,
    attribute: str,
    rng: np.random.Generator,
) -> DefensiveIdentity | None:
    if not defenders:
        return None
    weights = np.asarray(
        [
            max(defender.snap_weight, 0.001)
            * max(float(getattr(defender, attribute)), 0.10)
            for defender in defenders
        ],
        dtype=float,
    )
    weights /= weights.sum()
    return defenders[int(rng.choice(len(defenders), p=weights))]


def _sigmoid_margin(offense: float, defense: float, scale: float = 0.16) -> float:
    margin = (float(offense) - float(defense)) / max(scale, 1e-6)
    return float(1.0 / (1.0 + np.exp(-margin)))


def resolve_pass_interaction(
    *,
    quarterback: PlayerIdentity,
    target: PlayerIdentity,
    defense: DefensiveUnit,
    pass_protection: float,
    team_pass_efficiency: float,
    depth_category: str | None,
    rng: np.random.Generator,
) -> PassInteraction:
    """Resolve a target-specific coverage duel inside the full defensive structure.

    One coverage defender and one pressure defender are sampled from snap-weighted unit
    participation. The local duel is then blended with the surrounding unit so no single
    defender represents the entire defense. This creates genuine player-v-player variation
    while preserving 11-man context.
    """
    cover = _weighted_choice(defense.coverage, attribute="coverage", rng=rng)
    rusher = _weighted_choice(defense.front, attribute="pass_rush", rng=rng)

    cover_strength = 1.0 if cover is None else float(cover.coverage)
    rush_strength = 1.0 if rusher is None else float(rusher.pass_rush)
    ball_hawk = 1.0 if cover is None else float(cover.ball_hawk)

    route_skill = float(getattr(target, "route_separation", target.efficiency))
    receiver_speed = float(getattr(target, "speed_trait", target.explosive))
    qb_skill = float(getattr(quarterback, "qb_execution", quarterback.efficiency))
    qb_pressure = float(getattr(quarterback, "pressure_response", quarterback.efficiency))

    local_separation = _sigmoid_margin(route_skill, cover_strength)
    speed_separation = _sigmoid_margin(receiver_speed, cover_strength)
    depth_weight = {
        "screen": 0.15,
        "quick": 0.25,
        "short": 0.35,
        "intermediate": 0.55,
        "deep": 0.75,
        None: 0.45,
    }.get(depth_category, 0.45)
    separation = float(np.clip((1.0 - depth_weight) * local_separation + depth_weight * speed_separation, 0.08, 0.92))

    protection = max(float(pass_protection), 0.55)
    pressure_probability = float(
        np.clip(defense.pressure_rate * rush_strength / protection * (1.08 - 0.08 * qb_pressure), 0.08, 0.58)
    )
    completion_probability = float(
        np.clip(
            0.38
            + 0.34 * separation
            + 0.12 * qb_skill
            + 0.08 * team_pass_efficiency
            - 0.10 * max(cover_strength - 1.0, -0.35),
            0.24,
            0.90,
        )
    )
    interception_probability = float(
        np.clip(
            0.010
            + 0.030 * max(ball_hawk - 0.85, 0.0)
            + 0.018 * (1.0 - separation)
            - 0.010 * max(qb_skill - 1.0, -0.25),
            0.003,
            0.085,
        )
    )
    yards_multiplier = float(
        np.clip(0.72 + 0.70 * separation + 0.18 * (receiver_speed - 1.0), 0.55, 1.65)
    )
    return PassInteraction(
        coverage_defender_id=None if cover is None else cover.player_id,
        pressure_defender_id=None if rusher is None else rusher.player_id,
        separation=separation,
        pressure_probability=pressure_probability,
        completion_probability=completion_probability,
        interception_probability=interception_probability,
        yards_multiplier=yards_multiplier,
    )


def resolve_run_interaction(
    *,
    rusher: PlayerIdentity,
    defense: DefensiveUnit,
    run_blocking: float,
    run_geometry: str | None,
    rng: np.random.Generator,
) -> RunInteraction:
    """Resolve box penetration and second-level pursuit for one rushing attempt."""
    box = _weighted_choice(defense.front, attribute="run_defense", rng=rng)
    pursuit_pool = defense.coverage if defense.coverage else defense.front
    pursuit = _weighted_choice(pursuit_pool, attribute="tackling", rng=rng)

    box_strength = 1.0 if box is None else float(box.run_defense)
    tackle_strength = 1.0 if pursuit is None else float(pursuit.tackling)
    runner_creation = float(getattr(rusher, "rush_creation", rusher.efficiency))
    runner_power = float(getattr(rusher, "runner_power", rusher.efficiency))

    geometry_factor = {
        "interior": 1.10,
        "qb_sneak": 1.22,
        "left_offtackle": 1.02,
        "right_offtackle": 1.02,
        "left_edge": 0.92,
        "right_edge": 0.92,
        None: 1.0,
    }.get(run_geometry, 1.0)
    penetration_probability = float(
        np.clip(
            defense.run_stuff_rate * geometry_factor * box_strength / max(run_blocking, 0.55),
            0.05,
            0.48,
        )
    )
    creation_edge = _sigmoid_margin(runner_creation, box_strength)
    power_edge = _sigmoid_margin(runner_power, tackle_strength)
    yards_multiplier = float(
        np.clip(
            0.58
            + 0.62 * creation_edge
            + 0.34 * power_edge
            + 0.20 * (run_blocking - 1.0),
            0.48,
            1.72,
        )
    )
    return RunInteraction(
        box_defender_id=None if box is None else box.player_id,
        pursuit_defender_id=None if pursuit is None else pursuit.player_id,
        penetration_probability=penetration_probability,
        tackle_strength=tackle_strength,
        yards_multiplier=yards_multiplier,
    )
