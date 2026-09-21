from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality.game_latents import sample_game_day_latents, team_latent_vector
from monster.reality.latent_authority import LatentMechanism, route_team_latent
from monster.reality.world_state import GameDayLatents, WorldKey
from monster.reality.world_availability_v722 import materialize_world_pools_v722
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.player import TeamPlayerPool


_LATENT_CACHE: dict[tuple[str, int, int], GameDayLatents] = {}
_PREMISE_ROWS: list[dict[str, object]] = []
_RECORDED: set[tuple[str, int, int, str]] = set()


def reset_world_premise_v723() -> None:
    _LATENT_CACHE.clear()
    _PREMISE_ROWS.clear()
    _RECORDED.clear()


def materialize_world_pools_v723(
    pools: dict[str, TeamPlayerPool],
    *,
    seed: int,
    game: str,
) -> dict[str, TeamPlayerPool]:
    """Reuse V7.2.2 pregame availability; do not create a second role sampler."""

    return materialize_world_pools_v722(pools, seed=seed, game=game)


def _latents_for(*, seed: int, game: str, world: int) -> GameDayLatents:
    key = (str(game), int(world), int(seed))
    cached = _LATENT_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        away, home = str(game).split("@", 1)
    except ValueError as exc:
        raise ValueError("game must be '<away>@<home>'") from exc

    latents = sample_game_day_latents(
        WorldKey(game_id=str(game), world_index=int(world), seed=int(seed)),
        away_team_id=away,
        home_team_id=home,
    )
    _LATENT_CACHE[key] = latents
    return latents


def _bounded_multiplier(effective_value: float, *, low: float, high: float) -> float:
    return float(np.clip(np.exp(float(effective_value)), low, high))


def _player_with_execution(
    player: PlayerIdentity,
    *,
    efficiency_effect: float,
    explosive_effect: float = 0.0,
) -> PlayerIdentity:
    return replace(
        player,
        efficiency=float(
            np.clip(
                float(player.efficiency)
                * _bounded_multiplier(efficiency_effect, low=0.82, high=1.22),
                0.52,
                1.72,
            )
        ),
        explosive=float(
            np.clip(
                float(player.explosive)
                * _bounded_multiplier(explosive_effect, low=0.90, high=1.12),
                0.52,
                1.78,
            )
        ),
    )


def _regime(vector: dict[str, float]) -> tuple[str, float]:
    offense = (
        "qb_execution",
        "pass_protection",
        "receiver_execution",
        "run_blocking",
        "ballcarrier_execution",
    )
    cohesion = float(np.mean([float(vector.get(name, 0.0)) for name in offense]))
    if cohesion <= -0.80:
        label = "collapse"
    elif cohesion <= -0.30:
        label = "fragile"
    elif cohesion >= 0.80:
        label = "surge"
    else:
        label = "normal"
    return label, cohesion


def _routed(
    latents: GameDayLatents,
    *,
    team_id: str,
    factor: str,
    mechanism: LatentMechanism,
    authority: float,
) -> float:
    return route_team_latent(
        latents,
        team_id=team_id,
        factor=factor,
        mechanism=mechanism,
        requested_authority=authority,
    ).effective_value


def _record_premise(
    *,
    game: str,
    world: int,
    seed: int,
    team_id: str,
    latents: GameDayLatents,
) -> None:
    record_key = (str(game), int(world), int(seed), str(team_id))
    if record_key in _RECORDED:
        return
    _RECORDED.add(record_key)

    vector = team_latent_vector(latents, str(team_id))
    regime, cohesion = _regime(vector)
    _PREMISE_ROWS.append(
        {
            "game": str(game),
            "world": int(world),
            "seed": int(seed),
            "team": str(team_id),
            "regime": regime,
            "offensive_cohesion": cohesion,
            **{name: float(value) for name, value in vector.items()},
        }
    )


def materialize_world_team_v723(
    team: TeamIdentity,
    *,
    seed: int,
    game: str,
    world: int,
) -> TeamIdentity:
    """Route coherent game-day conditions into bounded local football mechanisms.

    This is epistemic world uncertainty above play RNG. It never adjusts score,
    fantasy points, salary, ownership, or optimizer output.
    """

    latents = _latents_for(seed=seed, game=game, world=world)
    team_id = str(team.team_id)
    _record_premise(
        game=game,
        world=world,
        seed=seed,
        team_id=team_id,
        latents=latents,
    )

    qb_effect = _routed(
        latents,
        team_id=team_id,
        factor="qb_execution",
        mechanism=LatentMechanism.THROW_EXECUTION,
        authority=0.10,
    )
    receiver_effect = _routed(
        latents,
        team_id=team_id,
        factor="receiver_execution",
        mechanism=LatentMechanism.CATCHPOINT,
        authority=0.10,
    )
    protection_effect = _routed(
        latents,
        team_id=team_id,
        factor="pass_protection",
        mechanism=LatentMechanism.PASS_PROTECTION,
        authority=0.08,
    )
    run_block_effect = _routed(
        latents,
        team_id=team_id,
        factor="run_blocking",
        mechanism=LatentMechanism.RUN_BLOCKING,
        authority=0.08,
    )
    ballcarrier_effect = _routed(
        latents,
        team_id=team_id,
        factor="ballcarrier_execution",
        mechanism=LatentMechanism.RUN_CONTACT,
        authority=0.10,
    )
    special_effect = _routed(
        latents,
        team_id=team_id,
        factor="special_teams_execution",
        mechanism=LatentMechanism.FIELD_GOAL,
        authority=0.05,
    )

    quarterback = _player_with_execution(
        team.quarterback,
        efficiency_effect=qb_effect,
    )
    receivers = tuple(
        _player_with_execution(
            player,
            efficiency_effect=receiver_effect,
            explosive_effect=0.45 * receiver_effect,
        )
        for player in team.receivers
    )
    rushers = tuple(
        _player_with_execution(
            player,
            efficiency_effect=ballcarrier_effect,
            explosive_effect=0.45 * ballcarrier_effect,
        )
        for player in team.rushers
    )

    return replace(
        team,
        quarterback=quarterback,
        receivers=receivers,
        rushers=rushers,
        pass_protection=float(
            np.clip(
                float(team.pass_protection)
                * _bounded_multiplier(protection_effect, low=0.86, high=1.16),
                0.74,
                1.28,
            )
        ),
        run_blocking=float(
            np.clip(
                float(team.run_blocking)
                * _bounded_multiplier(run_block_effect, low=0.86, high=1.16),
                0.74,
                1.28,
            )
        ),
        field_goal_skill=float(
            np.clip(
                float(team.field_goal_skill)
                * _bounded_multiplier(special_effect, low=0.92, high=1.08),
                0.82,
                1.18,
            )
        ),
        punt_skill=float(
            np.clip(
                float(team.punt_skill)
                * _bounded_multiplier(0.70 * special_effect, low=0.94, high=1.06),
                0.84,
                1.16,
            )
        ),
    )


def _defender_with_execution(
    player: DefensiveIdentity,
    *,
    pass_rush_effect: float,
    coverage_effect: float,
    run_fit_effect: float,
    tackling_effect: float,
) -> DefensiveIdentity:
    return replace(
        player,
        pass_rush=float(
            np.clip(
                float(player.pass_rush)
                * _bounded_multiplier(pass_rush_effect, low=0.84, high=1.18),
                0.50,
                1.72,
            )
        ),
        coverage=float(
            np.clip(
                float(player.coverage)
                * _bounded_multiplier(coverage_effect, low=0.84, high=1.18),
                0.50,
                1.72,
            )
        ),
        ball_hawk=float(
            np.clip(
                float(player.ball_hawk)
                * _bounded_multiplier(0.65 * coverage_effect, low=0.90, high=1.12),
                0.55,
                1.65,
            )
        ),
        run_defense=float(
            np.clip(
                float(player.run_defense)
                * _bounded_multiplier(run_fit_effect, low=0.84, high=1.18),
                0.50,
                1.72,
            )
        ),
        tackling=float(
            np.clip(
                float(player.tackling)
                * _bounded_multiplier(tackling_effect, low=0.86, high=1.16),
                0.52,
                1.68,
            )
        ),
    )


def materialize_world_defense_v723(
    defense: DefensiveUnit,
    *,
    team_id: str,
    seed: int,
    game: str,
    world: int,
) -> DefensiveUnit:
    latents = _latents_for(seed=seed, game=game, world=world)
    team_id = str(team_id)
    _record_premise(
        game=game,
        world=world,
        seed=seed,
        team_id=team_id,
        latents=latents,
    )

    pass_rush_effect = _routed(
        latents,
        team_id=team_id,
        factor="pass_rush",
        mechanism=LatentMechanism.PASS_RUSH,
        authority=0.08,
    )
    coverage_effect = _routed(
        latents,
        team_id=team_id,
        factor="coverage_execution",
        mechanism=LatentMechanism.COVERAGE,
        authority=0.08,
    )
    run_fit_effect = _routed(
        latents,
        team_id=team_id,
        factor="run_fit",
        mechanism=LatentMechanism.RUN_FIT,
        authority=0.08,
    )
    tackling_effect = _routed(
        latents,
        team_id=team_id,
        factor="tackling",
        mechanism=LatentMechanism.TACKLING,
        authority=0.08,
    )

    def transform(player: DefensiveIdentity) -> DefensiveIdentity:
        return _defender_with_execution(
            player,
            pass_rush_effect=pass_rush_effect,
            coverage_effect=coverage_effect,
            run_fit_effect=run_fit_effect,
            tackling_effect=tackling_effect,
        )

    return replace(
        defense,
        front=tuple(transform(player) for player in defense.front),
        coverage=tuple(transform(player) for player in defense.coverage),
        pressure_rate=float(
            np.clip(
                float(defense.pressure_rate)
                * _bounded_multiplier(0.45 * pass_rush_effect, low=0.93, high=1.08),
                0.12,
                0.50,
            )
        ),
        run_stuff_rate=float(
            np.clip(
                float(defense.run_stuff_rate)
                * _bounded_multiplier(0.45 * run_fit_effect, low=0.93, high=1.08),
                0.08,
                0.32,
            )
        ),
    )


def premise_rows_v723() -> list[dict[str, object]]:
    return list(_PREMISE_ROWS)


def write_world_premise_telemetry_v723(out: Path) -> None:
    if not _PREMISE_ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_PREMISE_ROWS).sort(["game", "world", "team"]).write_csv(
        out / "world_premise_v723.csv"
    )
