import polars as pl

from monster.feature_compile.player import compile_player_usage


def test_compile_player_usage_produces_shares():
    pbp = pl.DataFrame(
        {
            "posteam": ["A", "A", "A", "A", "A", "A"],
            "receiver_player_id": ["WR1", "WR1", "WR2", None, None, None],
            "rusher_player_id": [None, None, None, "RB1", "RB1", "QB1"],
            "pass_attempt": [1, 1, 1, 0, 0, 0],
            "rush_attempt": [0, 0, 0, 1, 1, 1],
            "yardline_100": [15, 50, 12, 10, 50, 8],
            "touchdown": [1, 0, 0, 1, 0, 0],
            "yards_gained": [12, 20, 8, 5, 11, 3],
            "air_yards": [9, 14, 7, None, None, None],
            "complete_pass": [1, 1, 1, 0, 0, 0],
            "pass_touchdown": [1, 0, 0, 0, 0, 0],
            "rush_touchdown": [0, 0, 0, 1, 0, 0],
        }
    )
    result = compile_player_usage(pbp)
    wr1 = result.filter(pl.col("player_id") == "WR1").row(0, named=True)
    rb1 = result.filter(pl.col("player_id") == "RB1").row(0, named=True)
    assert wr1["target_share"] > 0.6
    assert wr1["receiving_td_share"] == 1.0
    assert rb1["rush_share"] > 0.6
    assert rb1["rushing_td_share"] == 1.0
