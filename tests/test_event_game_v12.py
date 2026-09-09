from __future__ import annotations

import numpy as np

from monster.sim.event_game_v12 import simulate_event_game
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _team(team_id: str, opponent_id: str) -> TeamState:
    return TeamState(
        team_id=team_id,
        opponent_id=opponent_id,
        drives_per_game=10.7,
        offensive_epa_per_play=0.03,
        offensive_success_rate=0.45,
        offensive_explosive_rate=0.11,
        defensive_epa_allowed_per_play=0.01,
        defensive_explosive_rate_allowed=0.10,
        turnover_drive_rate=0.11,
        defensive_takeaway_drive_rate=0.11,
    )


def _pool(team_id: str) -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id=team_id,
        players=(
            PlayerState(player_id=f"{team_id}-QB", display_name="QB", position="QB", team_id=team_id, qb_pass_share=1.0, rush_share=0.10),
            PlayerState(player_id=f"{team_id}-RB", display_name="RB", position="RB", team_id=team_id, rush_share=0.60, target_share=0.12),
            PlayerState(player_id=f"{team_id}-WR", display_name="WR", position="WR", team_id=team_id, target_share=0.45),
        ),
        neutral_pass_rate=0.57,
        plays_per_drive=6.1,
        pass_td_share=0.64,
    )


def test_scoreboard_is_identity_over_simulated_scoring_events() -> None:
    game = GameState(game_id="A@H", away=_team("A", "H"), home=_team("H", "A"))
    worlds = simulate_event_game(game, _pool("A"), _pool("H"), worlds=4000, seed=1201)
    assert np.array_equal(worlds.away_points, 7 * worlds.away_touchdowns + 3 * worlds.away_field_goals)
    assert np.array_equal(worlds.home_points, 7 * worlds.home_touchdowns + 3 * worlds.home_field_goals)
    assert np.array_equal(worlds.away_touchdowns, worlds.away_passing_touchdowns + worlds.away_rushing_touchdowns)
    assert np.array_equal(worlds.home_touchdowns, worlds.home_passing_touchdowns + worlds.home_rushing_touchdowns)


def test_possessions_create_finite_plays_before_score() -> None:
    game = GameState(game_id="A@H", away=_team("A", "H"), home=_team("H", "A"))
    worlds = simulate_event_game(game, _pool("A"), _pool("H"), worlds=4000, seed=1202)
    assert worlds.away_plays is not None and worlds.home_plays is not None
    assert np.all(worlds.away_plays > 0)
    assert np.all(worlds.home_plays > 0)
    assert np.all(worlds.away_plays <= worlds.away_drives * 20)
    assert np.all(worlds.home_plays <= worlds.home_drives * 20)
    assert np.all(worlds.away_touchdowns <= worlds.away_drives)
    assert np.all(worlds.home_touchdowns <= worlds.home_drives)


def test_score_changes_only_when_scoring_events_exist() -> None:
    game = GameState(game_id="A@H", away=_team("A", "H"), home=_team("H", "A"))
    worlds = simulate_event_game(game, _pool("A"), _pool("H"), worlds=4000, seed=1203)
    away_no_score = (worlds.away_touchdowns == 0) & (worlds.away_field_goals == 0)
    home_no_score = (worlds.home_touchdowns == 0) & (worlds.home_field_goals == 0)
    assert np.all(worlds.away_points[away_no_score] == 0)
    assert np.all(worlds.home_points[home_no_score] == 0)
    assert worlds.away_touchdowns.sum() + worlds.home_touchdowns.sum() > 0
    assert worlds.away_field_goals.sum() + worlds.home_field_goals.sum() > 0
