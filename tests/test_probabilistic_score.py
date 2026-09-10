from __future__ import annotations

import numpy as np

from monster.audit.probabilistic_score import evaluate_score_worlds


def test_exact_observed_score_gets_positive_joint_probability() -> None:
    away = np.asarray([20, 24, 24, 27, 17], dtype=float)
    home = np.asarray([17, 21, 21, 20, 24], dtype=float)
    result = evaluate_score_worlds(away, home, actual_away=24, actual_home=21)
    assert result.exact_score_probability == 0.4
    assert result.nearby_score_probability_3 >= result.exact_score_probability
    assert result.nearby_score_probability_7 >= result.nearby_score_probability_3


def test_distribution_evaluation_preserves_win_probabilities() -> None:
    away = np.asarray([30, 20, 10, 24], dtype=float)
    home = np.asarray([20, 20, 17, 21], dtype=float)
    result = evaluate_score_worlds(away, home, actual_away=24, actual_home=21)
    assert result.away_win_probability == 0.5
    assert result.home_win_probability == 0.25
    assert result.tie_probability == 0.25
    assert result.winner_brier >= 0.0


def test_percentile_and_coverage_are_bounded() -> None:
    away = np.arange(10, 30, dtype=float)
    home = np.arange(15, 35, dtype=float)
    result = evaluate_score_worlds(away, home, actual_away=20, actual_home=25)
    assert 0.0 <= result.actual_total_percentile <= 1.0
    assert 0.0 <= result.actual_margin_percentile <= 1.0
    assert isinstance(result.central_90_total_covered, bool)
