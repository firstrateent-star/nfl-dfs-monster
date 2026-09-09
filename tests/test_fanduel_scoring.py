from __future__ import annotations

import numpy as np
import pytest

from monster.dfs.fanduel import score_offensive_player_worlds


def test_fanduel_scoring_exact_offensive_formula() -> None:
    stats = {
        "passing_yards": np.array([300.0]),
        "passing_tds": np.array([2.0]),
        "interceptions": np.array([1.0]),
        "rushing_yards": np.array([20.0]),
        "rushing_tds": np.array([1.0]),
        "receptions": np.array([4.0]),
        "receiving_yards": np.array([50.0]),
        "receiving_tds": np.array([1.0]),
        "fumbles_lost": np.array([1.0]),
    }
    # 12 + 8 - 1 + 2 + 6 + 2 + 5 + 6 - 2 = 38
    assert score_offensive_player_worlds(stats)[0] == pytest.approx(38.0)


def test_fanduel_scoring_refuses_missing_turnover_attribution() -> None:
    stats = {
        "passing_yards": np.array([250.0]),
        "passing_tds": np.array([2.0]),
        "rushing_yards": np.array([0.0]),
        "rushing_tds": np.array([0.0]),
        "receptions": np.array([0.0]),
        "receiving_yards": np.array([0.0]),
        "receiving_tds": np.array([0.0]),
    }
    with pytest.raises(ValueError, match="interceptions"):
        score_offensive_player_worlds(stats)
