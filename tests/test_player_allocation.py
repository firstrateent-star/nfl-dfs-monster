import numpy as np

from monster.sim.allocation import allocate_game_players
from monster.sim.game import simulate_game
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool(team: str) -> TeamPlayerPool:
    players = (
        PlayerState(
            f"{team}-QB",
            "QB One",
            "QB",
            team,
            rush_share=0.12,
            red_zone_rush_share=0.10,
            rushing_td_share=0.08,
            yards_per_carry=5.0,
            qb_pass_share=1.0,
        ),
        PlayerState(
            f"{team}-RB",
            "RB One",
            "RB",
            team,
            target_share=0.16,
            rush_share=0.72,
            red_zone_target_share=0.10,
            red_zone_rush_share=0.72,
            receiving_td_share=0.08,
            rushing_td_share=0.76,
            catch_rate=0.78,
            yards_per_reception=8.0,
            yards_per_carry=4.5,
        ),
        PlayerState(
            f"{team}-WR1",
            "WR One",
            "WR",
            team,
            target_share=0.50,
            red_zone_target_share=0.55,
            receiving_td_share=0.58,
            catch_rate=0.68,
            yards_per_reception=13.5,
            explosive_modifier=1.08,
        ),
        PlayerState(
            f"{team}-WR2",
            "WR Two",
            "WR",
            team,
            target_share=0.34,
            rush_share=0.16,
            red_zone_target_share=0.35,
            red_zone_rush_share=0.18,
            receiving_td_share=0.34,
            rushing_td_share=0.16,
            catch_rate=0.62,
            yards_per_reception=11.5,
        ),
    )
    return TeamPlayerPool(team, players, neutral_pass_rate=0.58, plays_per_drive=6.2)


def test_allocation_conserves_opportunities_and_touchdowns():
    away_state = TeamState("A", "H", td_drive_rate=0.26, fg_drive_rate=0.13)
    home_state = TeamState("H", "A", td_drive_rate=0.23, fg_drive_rate=0.14)
    worlds = simulate_game(GameState("A@H", away_state, home_state), 5000, 101)
    allocated = allocate_game_players(worlds, _pool("A"), _pool("H"), seed=202)

    for team in (allocated.away, allocated.home):
        target_sum = sum(stats["targets"] for stats in team.player_stats.values())
        rush_sum = sum(stats["rush_attempts"] for stats in team.player_stats.values())
        receiving_td_sum = sum(stats["receiving_tds"] for stats in team.player_stats.values())
        rushing_td_sum = sum(stats["rushing_tds"] for stats in team.player_stats.values())
        passing_td_sum = sum(stats["passing_tds"] for stats in team.player_stats.values())
        assert np.array_equal(target_sum, team.team_targets)
        assert np.array_equal(rush_sum, team.team_rush_attempts)
        assert np.array_equal(receiving_td_sum, team.passing_tds)
        assert np.array_equal(rushing_td_sum, team.rushing_tds)
        assert np.array_equal(passing_td_sum, team.passing_tds)


def test_role_share_changes_opportunity_distribution():
    state = TeamState("A", "H", td_drive_rate=0.25)
    worlds = simulate_game(GameState("A@H", state, state), 10000, 7)
    allocated = allocate_game_players(worlds, _pool("A"), _pool("H"), seed=8)
    wr1 = allocated.away.player_stats["A-WR1"]["targets"].mean()
    wr2 = allocated.away.player_stats["A-WR2"]["targets"].mean()
    rb = allocated.away.player_stats["A-RB"]["rush_attempts"].mean()
    qb = allocated.away.player_stats["A-QB"]["rush_attempts"].mean()
    assert wr1 > wr2
    assert rb > qb
