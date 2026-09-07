import numpy as np
from monster.snapshot.model import TeamState, GameState
from monster.sim.game import simulate_game


def test_sim_is_reproducible_and_nonnegative():
    a = TeamState("A", "B", .4, .1, .58, 1.0)
    b = TeamState("B", "A", .2, .3, .55, 1.0)
    g = GameState("A@B", a, b)
    x = simulate_game(g, 5000, 42)
    y = simulate_game(g, 5000, 42)
    assert np.array_equal(x.away_points, y.away_points)
    assert np.array_equal(x.home_points, y.home_points)
    assert np.array_equal(x.away_drives, y.away_drives)
    assert x.away_points.min() >= 0
    assert x.home_points.min() >= 0
    assert x.away_drives.min() >= 0
    assert x.home_drives.min() >= 0


def test_stronger_drive_scoring_priors_raise_scoring():
    weak = TeamState("W", "D", td_drive_rate=.13, fg_drive_rate=.10, offensive_epa_per_play=-.10)
    strong = TeamState("S", "D", td_drive_rate=.30, fg_drive_rate=.15, offensive_epa_per_play=.18)
    defense = TeamState("D", "S")
    weak_worlds = simulate_game(GameState("W@D", weak, defense), 20000, 9)
    strong_worlds = simulate_game(GameState("S@D", strong, defense), 20000, 9)
    assert strong_worlds.away_points.mean() > weak_worlds.away_points.mean()
