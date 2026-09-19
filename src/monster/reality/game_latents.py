from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.reality.world_state import GameDayLatents, LatentFactor, WorldKey


@dataclass(frozen=True)
class TeamLatentNames:
    team_id: str

    def name(self, mechanism: str) -> str:
        return f"{self.team_id}.{mechanism}"


_TEAM_MECHANISMS = (
    "qb_execution",
    "pass_protection",
    "receiver_execution",
    "run_blocking",
    "ballcarrier_execution",
    "pass_rush",
    "coverage_execution",
    "run_fit",
    "tackling",
    "special_teams_execution",
)


def _bounded(value: float) -> float:
    return float(np.clip(value, -2.75, 2.75))


def _team_factors(team_id: str, rng: np.random.Generator) -> tuple[LatentFactor, ...]:
    """Sample coherent mechanism-level conditions for one team.

    These are standardized latent conditions only. They have zero production
    authority until a specific football mechanism explicitly consumes one.
    Shared roots create realistic within-team co-movement without directly
    changing score, yards, fantasy points or role shares.
    """

    offense_day = float(rng.normal())
    pass_day = float(rng.normal())
    run_day = float(rng.normal())
    defense_day = float(rng.normal())
    pass_defense_day = float(rng.normal())
    run_defense_day = float(rng.normal())
    special_day = float(rng.normal())
    residual = rng.normal(size=len(_TEAM_MECHANISMS))

    values = {
        "qb_execution": 0.56 * offense_day + 0.56 * pass_day + 0.61 * residual[0],
        "pass_protection": 0.50 * offense_day + 0.52 * pass_day + 0.69 * residual[1],
        "receiver_execution": 0.48 * offense_day + 0.50 * pass_day + 0.72 * residual[2],
        "run_blocking": 0.52 * offense_day + 0.55 * run_day + 0.66 * residual[3],
        "ballcarrier_execution": 0.42 * offense_day + 0.52 * run_day + 0.74 * residual[4],
        "pass_rush": 0.55 * defense_day + 0.52 * pass_defense_day + 0.66 * residual[5],
        "coverage_execution": 0.52 * defense_day + 0.55 * pass_defense_day + 0.66 * residual[6],
        "run_fit": 0.56 * defense_day + 0.52 * run_defense_day + 0.65 * residual[7],
        "tackling": 0.62 * defense_day + 0.34 * run_defense_day + 0.70 * residual[8],
        "special_teams_execution": 0.72 * special_day + 0.69 * residual[9],
    }
    names = TeamLatentNames(team_id)
    return tuple(
        LatentFactor(
            name=names.name(mechanism),
            value=_bounded(values[mechanism]),
            source="v7_coherent_game_day_factor_model",
        )
        for mechanism in _TEAM_MECHANISMS
    )


def sample_game_day_latents(
    world: WorldKey,
    *,
    away_team_id: str,
    home_team_id: str,
) -> GameDayLatents:
    """Sample one deterministic latent performance world from the world seed.

    The random stream is deliberately separate from play-by-play RNG. This lets
    future paired counterfactuals hold the game-day world fixed while changing a
    single football mechanism.
    """

    if world.game_id != f"{away_team_id}@{home_team_id}":
        raise ValueError("team ids must match the world game id")

    # A fixed offset prevents future play-RNG refactors from silently changing
    # latent worlds while retaining reproducibility from the same WorldKey.
    seed_sequence = np.random.SeedSequence([world.seed, world.world_index, 7001])
    away_seed, home_seed = seed_sequence.spawn(2)
    factors = (
        *_team_factors(away_team_id, np.random.default_rng(away_seed)),
        *_team_factors(home_team_id, np.random.default_rng(home_seed)),
    )
    return GameDayLatents(factors)


def team_latent_vector(
    latents: GameDayLatents,
    team_id: str,
) -> dict[str, float]:
    prefix = f"{team_id}."
    return {
        factor.name.removeprefix(prefix): factor.value
        for factor in latents.factors
        if factor.name.startswith(prefix)
    }
