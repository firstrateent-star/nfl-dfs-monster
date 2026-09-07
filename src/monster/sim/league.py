from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import polars as pl

from monster.feature_compile.league_units import (
    compile_league_unit_effects,
    compile_league_unit_player_map,
)
from monster.feature_compile.mechanisms import PlayerMechanismInputs, TeamMechanismInputs
from monster.feature_compile.ol_simulation import compile_ol_simulation_context
from monster.registry import FeatureRegistry
from monster.sim.pipeline import MonsterGameWorlds, simulate_monster_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.snapshot.player import TeamPlayerPool


def simulate_league_context_game(
    *,
    game_id: str,
    away_team_id: str,
    home_team_id: str,
    policy: pl.DataFrame,
    personnel: pl.DataFrame,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    worlds: int,
    seed: int,
    team_inputs: Mapping[str, TeamMechanismInputs] | None = None,
    player_inputs: Mapping[str, PlayerMechanismInputs] | None = None,
    dome: bool = False,
    prior_uncertainty: float = 0.12,
    registry: FeatureRegistry | None = None,
) -> MonsterGameWorlds:
    """Assemble a game from canonical league context, then simulate football first.

    Current OL capability enters through the ordinary unit compiler. OL continuity and
    evidence coverage govern a separate sampled uncertainty state. Historical OL outcome
    signals are intentionally not accepted here as additive mean inputs because the team
    policy already contains overlapping historical pressure/rushing outcomes.
    """
    opponents = {away_team_id: home_team_id, home_team_id: away_team_id}
    states = compile_team_state_map(
        policy,
        opponents,
        prior_uncertainty=prior_uncertainty,
    )
    unit_map = compile_league_unit_player_map(personnel)
    unit_effects = compile_league_unit_effects(personnel)
    ol_context = compile_ol_simulation_context(personnel, unit_effects)
    ol_rows = {str(row["team_id"]): row for row in ol_context.to_dicts()}
    for team_id in (away_team_id, home_team_id):
        context = ol_rows.get(team_id)
        if context is not None:
            states[team_id] = replace(
                states[team_id],
                offensive_line_continuity=float(context["offensive_line_continuity"]),
                offensive_line_uncertainty=float(context["offensive_line_uncertainty"]),
            )

    if away_team_id not in unit_map or home_team_id not in unit_map:
        raise ValueError("League personnel artifact does not contain both game teams")
    if away_pool.team_id != away_team_id or home_pool.team_id != home_team_id:
        raise ValueError("Skill-player pool team IDs must match requested game teams")

    mechanisms = team_inputs or {}
    game = GameState(
        game_id=game_id,
        away=states[away_team_id],
        home=states[home_team_id],
        dome=dome,
        feature_names=frozenset(
            {
                "historical_team_policy",
                "league_personnel_participation",
                "observed_unit_capability",
                "offensive_line_continuity_uncertainty",
            }
        ),
    )
    return simulate_monster_game(
        game,
        away_pool,
        home_pool,
        worlds=worlds,
        seed=seed,
        away_team_inputs=mechanisms.get(away_team_id),
        home_team_inputs=mechanisms.get(home_team_id),
        away_unit_players=unit_map[away_team_id],
        home_unit_players=unit_map[home_team_id],
        player_inputs=player_inputs,
        registry=registry,
    )
