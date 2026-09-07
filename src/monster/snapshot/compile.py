from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    PlayerMechanismTrace,
    TeamMechanismInputs,
    TeamMechanismTrace,
    compile_player_mechanisms,
    compile_team_mechanisms,
)
from monster.registry import FeatureRegistry
from monster.snapshot.guard import assert_market_blind
from monster.snapshot.model import GameState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class CompiledGameSnapshot:
    game: GameState
    away_pool: TeamPlayerPool
    home_pool: TeamPlayerPool
    player_traces: dict[str, PlayerMechanismTrace]
    team_traces: dict[str, TeamMechanismTrace]


def _compile_pool(
    pool: TeamPlayerPool,
    player_inputs: Mapping[str, PlayerMechanismInputs],
) -> tuple[TeamPlayerPool, dict[str, PlayerMechanismTrace]]:
    compiled_players = []
    traces: dict[str, PlayerMechanismTrace] = {}
    for player in pool.players:
        compiled, trace = compile_player_mechanisms(
            player,
            player_inputs.get(player.player_id, PlayerMechanismInputs()),
        )
        compiled_players.append(compiled)
        traces[player.player_id] = trace
    return replace(pool, players=tuple(compiled_players)), traces


def compile_game_snapshot(
    game: GameState,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    away_team_inputs: TeamMechanismInputs | None = None,
    home_team_inputs: TeamMechanismInputs | None = None,
    player_inputs: Mapping[str, PlayerMechanismInputs] | None = None,
    registry: FeatureRegistry | None = None,
) -> CompiledGameSnapshot:
    """Compile a market-blind game snapshot immediately before simulation.

    The database/source layer may contain many raw facts. This function is the
    narrow seam that turns only approved football features into simulation state.
    """
    if registry is not None:
        assert_market_blind(set(game.feature_names), registry)

    player_inputs = player_inputs or {}
    away_pool, away_player_traces = _compile_pool(away_pool, player_inputs)
    home_pool, home_player_traces = _compile_pool(home_pool, player_inputs)

    away_team, away_pool, away_team_trace = compile_team_mechanisms(
        game.away,
        away_pool,
        away_team_inputs or TeamMechanismInputs(dome=game.dome),
    )
    home_team, home_pool, home_team_trace = compile_team_mechanisms(
        game.home,
        home_pool,
        home_team_inputs or TeamMechanismInputs(dome=game.dome),
    )

    compiled_game = replace(game, away=away_team, home=home_team)
    traces = {**away_player_traces, **home_player_traces}
    return CompiledGameSnapshot(
        game=compiled_game,
        away_pool=away_pool,
        home_pool=home_pool,
        player_traces=traces,
        team_traces={
            away_team.team_id: away_team_trace,
            home_team.team_id: home_team_trace,
        },
    )
