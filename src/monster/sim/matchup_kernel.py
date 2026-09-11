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
    local_coverage_strength: float = 1.0
    local_separation_edge: float = 0.0
    local_rush_strength: float = 1.0


@dataclass(frozen=True)
class RunMatchup:
    stuff_probability: float
    yards_multiplier: float
    primary_defender_id: str | None
    local_run_defense: float = 1.0
    second_level_tackling: float = 1.0
    runner_edge: float = 0.0


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
    """Select the most likely local matchup participant from role x snap x skill."""
    if not defenders:
        return None
    return max(
        defenders,
        key=lambda defender: max(defender.snap_weight, 0.001)
        * max(float(getattr(defender, attribute)), 0.001),
    )


def _duel_edge(offense: float, defense: float, *, scale: float = 0.24) -> float:
    """Bounded -1..1 local player-v-player advantage."""
    return float(np.tanh((float(offense) - float(defense)) / max(scale, 1e-6)))


def resolve_pass_matchup(
    target: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    pass_protection: float,
    quarterback_efficiency: float,
) -> PassMatchup:
    """Resolve receiver-v-cover and protection-v-rush within the full defense.

    The target's own skill is compared directly with the most exposed/likely coverage
    defender. That 1v1 edge is blended with the exposure-weighted coverage unit, while an
    individual front defender supplies the local rush threat inside the front's aggregate
    pressure context. This makes player identity causal without pretending one duel is the
    entire 11-v-11 play.
    """
    cover = _representative_defender(defense.coverage, "coverage")
    rusher = _representative_defender(defense.front, "pass_rush")
    coverage_unit = _unit_strength(defense.coverage, "coverage")
    ball_hawk_unit = _unit_strength(defense.coverage, "ball_hawk")
    rush_unit = _unit_strength(defense.front, "pass_rush")

    local_coverage = 1.0 if cover is None else float(cover.coverage)
    local_ball_hawk = 1.0 if cover is None else float(cover.ball_hawk)
    local_rush = 1.0 if rusher is None else float(rusher.pass_rush)
    separation_edge = _duel_edge(target.efficiency, local_coverage)

    # Local edge has substantial authority, but surrounding help/structure remains causal.
    effective_coverage = float(
        np.clip(0.58 * coverage_unit + 0.42 * local_coverage - 0.10 * separation_edge, 0.55, 1.45)
    )
    effective_ball_hawk = float(np.clip(0.65 * ball_hawk_unit + 0.35 * local_ball_hawk, 0.55, 1.45))
    effective_rush = float(np.clip(0.62 * rush_unit + 0.38 * local_rush, 0.55, 1.50))

    pressure = float(
        np.clip(defense.pressure_rate * effective_rush / max(pass_protection, 0.55), 0.10, 0.55)
    )
    completion = float(
        np.clip(
            0.64
            * quarterback_efficiency
            * target.efficiency
            / max(effective_coverage, 0.60)
            * (1.0 + 0.06 * separation_edge),
            0.30,
            0.88,
        )
    )
    interception = float(
        np.clip(
            0.022
            * effective_ball_hawk
            * effective_coverage
            / max(quarterback_efficiency, 0.55)
            * (1.0 - 0.10 * separation_edge),
            0.004,
            0.08,
        )
    )
    yards_multiplier = float(
        np.clip(
            target.explosive
            / max(effective_coverage**0.35, 0.72)
            * (1.0 + 0.10 * separation_edge),
            0.60,
            1.65,
        )
    )
    return PassMatchup(
        pressure_probability=pressure,
        interception_probability=interception,
        completion_probability=completion,
        yards_multiplier=yards_multiplier,
        primary_defender_id=None if cover is None else cover.player_id,
        coverage_strength=effective_coverage,
        ball_hawk_strength=effective_ball_hawk,
        local_coverage_strength=local_coverage,
        local_separation_edge=separation_edge,
        local_rush_strength=local_rush,
    )


def resolve_run_matchup(
    rusher: PlayerIdentity,
    defense: DefensiveUnit,
    *,
    run_blocking: float,
) -> RunMatchup:
    """Resolve rusher-v-box contact inside front and pursuit context."""
    box = _representative_defender(defense.front, "run_defense")
    pursuit_pool = defense.coverage if defense.coverage else defense.front
    pursuit = _representative_defender(pursuit_pool, "tackling")
    front_unit = _unit_strength(defense.front, "run_defense")
    tackling_unit = _unit_strength(pursuit_pool, "tackling")

    local_run_defense = 1.0 if box is None else float(box.run_defense)
    local_tackling = 1.0 if pursuit is None else float(pursuit.tackling)
    runner_edge = _duel_edge(rusher.efficiency, local_run_defense)

    effective_front = float(
        np.clip(0.60 * front_unit + 0.40 * local_run_defense - 0.08 * runner_edge, 0.55, 1.50)
    )
    second_level = float(np.clip(0.55 * tackling_unit + 0.45 * local_tackling, 0.55, 1.50))
    stuff = float(
        np.clip(
            defense.run_stuff_rate
            * effective_front
            / max(run_blocking, 0.55)
            * (1.0 - 0.08 * runner_edge),
            0.05,
            0.46,
        )
    )
    yards_multiplier = float(
        np.clip(
            rusher.efficiency
            * run_blocking
            / max(effective_front, 0.60)
            * (1.0 + 0.08 * runner_edge)
            / max(second_level**0.12, 0.92),
            0.50,
            1.68,
        )
    )
    return RunMatchup(
        stuff_probability=stuff,
        yards_multiplier=yards_multiplier,
        primary_defender_id=None if box is None else box.player_id,
        local_run_defense=local_run_defense,
        second_level_tackling=second_level,
        runner_edge=runner_edge,
    )
