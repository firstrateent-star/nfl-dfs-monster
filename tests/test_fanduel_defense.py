import numpy as np

from monster.dfs.defense import points_allowed_score, score_defense_partial_worlds


def test_points_allowed_bands() -> None:
    points = np.array([0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 50])
    expected = np.array([10, 7, 7, 4, 4, 1, 1, 0, 0, -1, -1, -4, -4], dtype=np.float32)
    np.testing.assert_array_equal(points_allowed_score(points), expected)


def test_partial_defense_uses_same_world_takeaways() -> None:
    points = np.array([0, 17, 31, 42])
    turnovers = np.array([1, 2, 0, 3])
    np.testing.assert_array_equal(
        score_defense_partial_worlds(opponent_points=points, opponent_turnovers=turnovers),
        np.array([12, 5, -1, 2], dtype=np.float32),
    )


def test_partial_defense_rejects_misaligned_worlds() -> None:
    try:
        score_defense_partial_worlds(
            opponent_points=np.array([10, 20]), opponent_turnovers=np.array([1])
        )
    except ValueError as exc:
        assert "correlated world shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to fail")
