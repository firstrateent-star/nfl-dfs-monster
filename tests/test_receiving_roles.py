import numpy as np

from monster.sim.allocation import allocate_game_players
from monster.sim.game import simulate_game
from monster.sim.receiving_roles import refine_game_receiving_roles
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool(team: str) -> TeamPlayerPool:
    target_shares = (0.27, 0.20, 0.15, 0.12, 0.09, 0.07, 0.05, 0.05)
    players = [
        PlayerState(
            f"{team}-QB",
            "QB One",
            "QB",
            team,
            qb_pass_share=1.0,
            rush_share=0.10,
            yards_per_carry=4.8,
        )
    ]
    for idx, share in enumerate(target_shares, start=1):
        players.append(
            PlayerState(
                f"{team}-R{idx}",
                f"Receiver {idx}",
                "WR" if idx <= 5 else "TE",
                team,
                target_share=share,
                receiving_td_share=share,
                red_zone_target_share=share,
                catch_rate=0.66,
                yards_per_reception=11.0,
                role_uncertainty=0.08,
            )
        )
    return TeamPlayerPool(team, tuple(players), neutral_pass_rate=0.60, plays_per_drive=6.2)


def test_receiving_refinement_conserves_finite_supply_and_qb_accounting():
    away_pool = _pool("A")
    home_pool = _pool("H")
    away_state = TeamState("A", "H", td_drive_rate=0.26, fg_drive_rate=0.13)
    home_state = TeamState("H", "A", td_drive_rate=0.23, fg_drive_rate=0.14)
    game = simulate_game(GameState("A@H", away_state, home_state), 4000, 101)
    allocated = allocate_game_players(game, away_pool, home_pool, seed=202)
    refined = refine_game_receiving_roles(
        allocated, away_pool, home_pool, seed=303
    )

    for side in (refined.away, refined.home):
        target_sum = sum(stats["targets"] for stats in side.player_stats.values())
        receiving_td_sum = sum(stats["receiving_tds"] for stats in side.player_stats.values())
        reception_sum = sum(stats["receptions"] for stats in side.player_stats.values())
        qb_completion_sum = sum(
            stats["completions"]
            for player_id, stats in side.player_stats.items()
            if player_id.endswith("-QB")
        )
        passing_td_sum = sum(
            stats["passing_tds"]
            for player_id, stats in side.player_stats.items()
            if player_id.endswith("-QB")
        )
        assert np.array_equal(target_sum, side.team_targets)
        assert np.array_equal(receiving_td_sum, side.passing_tds)
        assert np.array_equal(qb_completion_sum, reception_sum)
        assert np.array_equal(passing_td_sum, side.passing_tds)


def test_receiving_refinement_creates_a_finite_ranked_tree_not_one_star_monopoly():
    pool = _pool("A")
    state = TeamState("A", "H", td_drive_rate=0.25)
    game = simulate_game(GameState("A@H", state, state), 6000, 707)
    allocated = allocate_game_players(game, pool, _pool("H"), seed=808)
    refined = refine_game_receiving_roles(allocated, pool, _pool("H"), seed=909)

    matrix = np.column_stack(
        [stats["targets"] for stats in refined.away.player_stats.values()]
    )
    ranked = np.sort(matrix, axis=1)[:, ::-1]
    total = np.maximum(refined.away.team_targets.astype(float), 1.0)
    leader_share = float(np.mean(ranked[:, 0] / total))
    second_share = float(np.mean(ranked[:, 1] / total))
    earners = float(np.mean((matrix > 0).sum(axis=1)))

    assert 0.24 <= leader_share <= 0.36
    assert 0.16 <= second_share <= 0.27
    assert earners >= 6.0
