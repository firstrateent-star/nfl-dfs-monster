from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    TeamMechanismInputs,
)
from monster.feature_compile.units import UnitPlayerInputs
from monster.registry import FeatureRegistry
from monster.sim.allocation import GameAllocationWorlds, allocate_game_players
from monster.sim.game import GameWorlds, simulate_game
from monster.sim.rushing_roles import refine_game_rushing_roles
from monster.snapshot.compile import CompiledGameSnapshot, compile_game_snapshot
from monster.snapshot.model import GameState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class MonsterGameWorlds:
    snapshot: CompiledGameSnapshot
    game_worlds: GameWorlds
    allocation_worlds: GameAllocationWorlds


def simulate_monster_game(
    game: GameState,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    worlds: int,
    seed: int,
    away_team_inputs: TeamMechanismInputs | None = None,
    home_team_inputs: TeamMechanismInputs | None = None,
    away_unit_players: tuple[UnitPlayerInputs, ...] | None = None,
    home_unit_players: tuple[UnitPlayerInputs, ...] | None = None,
    player_inputs: Mapping[str, PlayerMechanismInputs] | None = None,
    registry: FeatureRegistry | None = None,
) -> MonsterGameWorlds:
    """Compile complete football personnel, simulate scoring, then allocate supply.

    The scoring state is informed by snap-weighted offense, OL, defense and special
    teams personnel before fantasy-relevant player opportunities are allocated.
    """
    snapshot = compile_game_snapshot(
        game,
        away_pool,
        home_pool,
        away_team_inputs=away_team_inputs,
        home_team_inputs=home_team_inputs,
        away_unit_players=away_unit_players,
        home_unit_players=home_unit_players,
        player_inputs=player_inputs,
        registry=registry,
    )
    game_worlds = simulate_game(snapshot.game, worlds=worlds, seed=seed)
    allocation_worlds = allocate_game_players(
        game_worlds,
        snapshot.away_pool,
        snapshot.home_pool,
        seed=seed + 1_000_003,
    )
    allocation_worlds = refine_game_rushing_roles(
        allocation_worlds,
        snapshot.away_pool,
        snapshot.home_pool,
        seed=seed + 2_000_033,
    )
    return MonsterGameWorlds(
        snapshot=snapshot,
        game_worlds=game_worlds,
        allocation_worlds=allocation_worlds,
    )
