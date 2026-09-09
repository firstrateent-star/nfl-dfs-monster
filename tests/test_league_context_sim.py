from __future__ import annotations

import polars as pl

from monster.sim.league import simulate_league_context_game
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool(team: str) -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id=team,
        players=(
            PlayerState(
                player_id=f"{team}-qb",
                display_name=f"{team} QB",
                position="QB",
                team_id=team,
                qb_pass_share=1.0,
            ),
            PlayerState(
                player_id=f"{team}-rb",
                display_name=f"{team} RB",
                position="RB",
                team_id=team,
                rush_share=1.0,
                rushing_td_share=1.0,
            ),
            PlayerState(
                player_id=f"{team}-wr",
                display_name=f"{team} WR",
                position="WR",
                team_id=team,
                target_share=1.0,
                receiving_td_share=1.0,
            ),
        ),
    )


def test_league_context_simulation_assembles_policy_and_unit_personnel():
    policy = pl.DataFrame(
        {
            "team_id": ["A", "B"],
            "neutral_pass_rate": [0.58, 0.54],
            "drives_per_game": [10.8, 10.4],
            "td_drive_rate": [0.24, 0.21],
            "fg_drive_rate": [0.13, 0.14],
            "turnover_drive_rate": [0.10, 0.12],
            "red_zone_td_rate": [0.58, 0.53],
        }
    )
    personnel = pl.DataFrame(
        {
            "team_id": ["A", "A", "B", "B"],
            "gsis_id": ["a-dl", "a-db", "b-dl", "b-db"],
            "position": ["DE", "CB", "DE", "CB"],
            "projected_offense_snap_share": [0.0, 0.0, 0.0, 0.0],
            "projected_defense_snap_share": [0.6, 0.4, 0.6, 0.4],
            "projected_special_teams_snap_share": [0.0, 0.0, 0.0, 0.0],
            "participation_uncertainty": [0.10, 0.10, 0.10, 0.10],
            "observed_pass_rush_signal": [0.5, None, -0.2, None],
            "observed_coverage_signal": [None, 0.4, None, -0.1],
            "observed_run_defense_signal": [0.3, 0.1, 0.0, -0.1],
        }
    )
    result = simulate_league_context_game(
        game_id="A-B",
        away_team_id="A",
        home_team_id="B",
        policy=policy,
        personnel=personnel,
        away_pool=_pool("A"),
        home_pool=_pool("B"),
        worlds=250,
        seed=7,
    )
    assert result.game_worlds.away_points.shape == (250,)
    assert result.game_worlds.home_points.shape == (250,)
    assert result.snapshot.game.away.pass_rush_effect > 0.0
    assert result.snapshot.game.home.pass_rush_effect < 0.0
