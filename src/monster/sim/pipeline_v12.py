from __future__ import annotations

from collections.abc import Mapping

from monster.feature_compile.mechanisms import PlayerMechanismInputs, TeamMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.registry import FeatureRegistry
from monster.sim.event_allocation_v12 import allocate_event_game_players
from monster.sim.event_coupling import couple_touchdowns_to_events
from monster.sim.event_game_v12 import simulate_event_game
from monster.sim.pipeline import MonsterGameWorlds
from monster.sim.receiving_roles import refine_game_receiving_roles
from monster.sim.rushing_roles import refine_game_rushing_roles
from monster.snapshot.compile import compile_game_snapshot
from monster.snapshot.model import GameState
from monster.snapshot.player import TeamPlayerPool


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
    """Monster v1.2: simulate football events first and infer the scoreboard from them."""
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
    game_worlds = simulate_event_game(snapshot.game, snapshot.away_pool, snapshot.home_pool, worlds=worlds, seed=seed)
    allocation = allocate_event_game_players(game_worlds, snapshot.away_pool, snapshot.home_pool, seed=seed + 1_000_003)
    allocation = refine_game_receiving_roles(allocation, snapshot.away_pool, snapshot.home_pool, seed=seed + 2_000_021)
    allocation = refine_game_rushing_roles(allocation, snapshot.away_pool, snapshot.home_pool, seed=seed + 3_000_033)
    # Couple scoring ownership last. Role refiners may redistribute targets/carries, but they may
    # never leave a TD attached to a player who did not produce its source reception/carry.
    allocation = couple_touchdowns_to_events(
        allocation,
        snapshot.away_pool,
        snapshot.home_pool,
        seed=seed + 4_000_037,
    )

    for side, game_plays, pass_tds, rush_tds in (
        (allocation.away, game_worlds.away_plays, game_worlds.away_passing_touchdowns, game_worlds.away_rushing_touchdowns),
        (allocation.home, game_worlds.home_plays, game_worlds.home_passing_touchdowns, game_worlds.home_rushing_touchdowns),
    ):
        if game_plays is None or pass_tds is None or rush_tds is None:
            raise AssertionError("v1.2 event supply disappeared during refinement")
        if not (side.team_plays == game_plays).all():
            raise AssertionError("player refinement changed finite team play supply")
        if not (side.passing_tds == pass_tds).all():
            raise AssertionError("player refinement changed passing touchdown supply")
        if not (side.rushing_tds == rush_tds).all():
            raise AssertionError("player refinement changed rushing touchdown supply")
    return MonsterGameWorlds(snapshot=snapshot, game_worlds=game_worlds, allocation_worlds=allocation)
