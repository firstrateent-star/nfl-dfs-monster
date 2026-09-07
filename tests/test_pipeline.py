import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.sim.pipeline import simulate_monster_game
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool(team_id: str) -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id=team_id,
        players=(
            PlayerState(
                player_id=f"{team_id}-qb",
                display_name=f"{team_id} QB",
                position="QB",
                team_id=team_id,
                qb_pass_share=1.0,
                rush_share=0.12,
            ),
            PlayerState(
                player_id=f"{team_id}-wr",
                display_name=f"{team_id} WR",
                position="WR",
                team_id=team_id,
                target_share=0.62,
                receiving_td_share=0.58,
                catch_rate=0.68,
                yards_per_reception=12.0,
            ),
            PlayerState(
                player_id=f"{team_id}-rb",
                display_name=f"{team_id} RB",
                position="RB",
                team_id=team_id,
                target_share=0.18,
                rush_share=0.70,
                rushing_td_share=0.72,
                yards_per_carry=4.4,
            ),
        ),
    )


def _game() -> GameState:
    return GameState(
        "A@H",
        TeamState("A", "H", td_drive_rate=0.25),
        TeamState("H", "A", td_drive_rate=0.23),
    )


def test_extra_speed_changes_player_mechanism_not_team_score_worlds():
    neutral = simulate_monster_game(
        _game(),
        _pool("A"),
        _pool("H"),
        worlds=12000,
        seed=77,
    )
    fast = simulate_monster_game(
        _game(),
        _pool("A"),
        _pool("H"),
        worlds=12000,
        seed=77,
        player_inputs={
            "A-wr": PlayerMechanismInputs(
                forty_time=4.28,
                madden_speed=99.0,
                madden_acceleration=98.0,
            )
        },
    )
    assert np.array_equal(neutral.game_worlds.away_points, fast.game_worlds.away_points)
    neutral_yards = neutral.allocation_worlds.away.player_stats["A-wr"]["receiving_yards"]
    fast_yards = fast.allocation_worlds.away.player_stats["A-wr"]["receiving_yards"]
    assert fast_yards.mean() > neutral_yards.mean()


def test_astrology_shadow_cannot_change_production_worlds():
    negative = simulate_monster_game(
        _game(),
        _pool("A"),
        _pool("H"),
        worlds=5000,
        seed=88,
        player_inputs={"A-wr": PlayerMechanismInputs(astrology_shadow_signal=-1.0)},
    )
    positive = simulate_monster_game(
        _game(),
        _pool("A"),
        _pool("H"),
        worlds=5000,
        seed=88,
        player_inputs={"A-wr": PlayerMechanismInputs(astrology_shadow_signal=1.0)},
    )
    assert np.array_equal(negative.game_worlds.away_points, positive.game_worlds.away_points)
    for stat_name, negative_values in negative.allocation_worlds.away.player_stats["A-wr"].items():
        positive_values = positive.allocation_worlds.away.player_stats["A-wr"][stat_name]
        assert np.array_equal(negative_values, positive_values), stat_name
