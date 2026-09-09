from __future__ import annotations

from collections.abc import Mapping

from monster.feature_compile.mechanisms import PlayerMechanismInputs, TeamMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.registry import FeatureRegistry
from monster.sim.event_coupling import couple_touchdowns_to_events
from monster.sim.pipeline import MonsterGameWorlds
from monster.sim.pipeline import simulate_monster_game as simulate_v1
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
    """Monster v1.1 seam: certified v1 football plus event-coupled player TD outcomes."""
    result = simulate_v1(
        game,
        away_pool,
        home_pool,
        worlds=worlds,
        seed=seed,
        away_team_inputs=away_team_inputs,
        home_team_inputs=home_team_inputs,
        away_unit_players=away_unit_players,
        home_unit_players=home_unit_players,
        player_inputs=player_inputs,
        registry=registry,
    )
    allocation = couple_touchdowns_to_events(
        result.allocation_worlds,
        result.snapshot.away_pool,
        result.snapshot.home_pool,
        seed=seed + 4_000_037,
    )
    return MonsterGameWorlds(
        snapshot=result.snapshot,
        game_worlds=result.game_worlds,
        allocation_worlds=allocation,
    )
