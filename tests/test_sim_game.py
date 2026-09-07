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
    assert x.away_points.min() >= 0
    assert x.home_points.min() >= 0
