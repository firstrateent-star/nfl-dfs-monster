import numpy as np
import pytest

from monster.dfs.draftkings import score_offensive_player_worlds


def _stats() -> dict[str, np.ndarray]:
    return {
        "passing_yards": np.array([299.0, 300.0]),
        "passing_tds": np.array([2.0, 2.0]),
        "passing_interceptions": np.array([1.0, 0.0]),
        "rushing_yards": np.array([99.0, 100.0]),
        "rushing_tds": np.array([0.0, 1.0]),
        "receptions": np.array([4.0, 5.0]),
        "receiving_yards": np.array([99.0, 100.0]),
        "receiving_tds": np.array([1.0, 1.0]),
        "fumbles_lost": np.array([1.0, 0.0]),
    }


def test_draftkings_scoring_applies_world_level_yardage_bonuses() -> None:
    points = score_offensive_player_worlds(_stats())
    expected_first = 299 * 0.04 + 2 * 4 - 1 + 99 * 0.1 + 4 + 99 * 0.1 + 6 - 1
    expected_second = 300 * 0.04 + 2 * 4 + 100 * 0.1 + 6 + 5 + 100 * 0.1 + 6 + 9
    assert points[0] == pytest.approx(expected_first)
    assert points[1] == pytest.approx(expected_second)


def test_draftkings_scoring_rejects_missing_football_state() -> None:
    stats = _stats()
    del stats["fumbles_lost"]
    with pytest.raises(ValueError, match="fumbles_lost"):
        score_offensive_player_worlds(stats)
