from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    PlayerMechanismTrace,
    TeamMechanismInputs,
    TeamMechanismTrace,
    compile_player_mechanisms,
    compile_team_mechanisms,
)
from monster.feature_compile.units import (
    UnitPlayerInputs,
    UnitTrace,
    apply_team_unit_effects,
    compile_team_unit_effects,
)
from monster.registry import FeatureRegistry
from monster.snapshot.guard import assert_market_blind
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class CompiledGameSnapshot:
    game: GameState
    away_pool: TeamPlayerPool
    home_pool: TeamPlayerPool
    player_traces: dict[str, PlayerMechanismTrace]
    team_traces: dict[str, TeamMechanismTrace]
    unit_traces: dict[str, UnitTrace]


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


def _neutral_team_trace(team: TeamState) -> TeamMechanismTrace:
    return TeamMechanismTrace(
        personnel_signal=0.0,
        weather_effect=0.0,
        continuity_signal=0.0,
        coaching_entropy=team.coaching_entropy,
    )


def _neutral_unit_trace() -> UnitTrace:
    return UnitTrace(
        offense_snap_weight=0.0,
        defense_snap_weight=0.0,
        special_teams_snap_weight=0.0,
        players_with_offense_weight=0,
        players_with_defense_weight=0,
        players_with_special_teams_weight=0,
    )


def _apply_unit_personnel(
    team: TeamState,
    players: tuple[UnitPlayerInputs, ...] | None,
) -> tuple[TeamState, UnitTrace]:
    if not players:
        return team, _neutral_unit_trace()
    effects, trace = compile_team_unit_effects(players)
    return apply_team_unit_effects(team, effects), trace


def compile_game_snapshot(
    game: GameState,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    away_team_inputs: TeamMechanismInputs | None = None,
    home_team_inputs: TeamMechanismInputs | None = None,
    away_unit_players: tuple[UnitPlayerInputs, ...] | None = None,
    home_unit_players: tuple[UnitPlayerInputs, ...] | None = None,
    player_inputs: Mapping[str, PlayerMechanismInputs] | None = None,
    registry: FeatureRegistry | None = None,
) -> CompiledGameSnapshot:
    """Compile the complete market-blind football snapshot before simulation.

    Fantasy-relevant skill players are only one subset of the game. The unit layer
    aggregates offense, offensive line, defense and special-teams personnel according
    to projected snap shares, then attaches those mechanisms to the same team state
    used by the drive simulator.
    """
    if registry is not None:
        assert_market_blind(set(game.feature_names), registry)

    player_inputs = player_inputs or {}
    away_pool, away_player_traces = _compile_pool(away_pool, player_inputs)
    home_pool, home_player_traces = _compile_pool(home_pool, player_inputs)

    away_team, away_unit_trace = _apply_unit_personnel(game.away, away_unit_players)
    home_team, home_unit_trace = _apply_unit_personnel(game.home, home_unit_players)

    if away_team_inputs is None:
        away_team_trace = _neutral_team_trace(away_team)
    else:
        away_team, away_pool, away_team_trace = compile_team_mechanisms(
            away_team,
            away_pool,
            away_team_inputs,
        )

    if home_team_inputs is None:
        home_team_trace = _neutral_team_trace(home_team)
    else:
        home_team, home_pool, home_team_trace = compile_team_mechanisms(
            home_team,
            home_pool,
            home_team_inputs,
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
        unit_traces={
            away_team.team_id: away_unit_trace,
            home_team.team_id: home_unit_trace,
        },
    )
