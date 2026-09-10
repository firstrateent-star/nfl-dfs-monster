from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreDistributionEvaluation:
    actual_away: int
    actual_home: int
    away_mean: float
    home_mean: float
    total_mean: float
    margin_mean: float
    away_mae: float
    home_mae: float
    total_absolute_error: float
    margin_absolute_error: float
    actual_total_percentile: float
    actual_margin_percentile: float
    actual_away_percentile: float
    actual_home_percentile: float
    central_50_total_covered: bool
    central_80_total_covered: bool
    central_90_total_covered: bool
    central_50_margin_covered: bool
    central_80_margin_covered: bool
    central_90_margin_covered: bool
    away_win_probability: float
    home_win_probability: float
    tie_probability: float
    winner_brier: float
    exact_score_probability: float
    nearby_score_probability_3: float
    nearby_score_probability_7: float


def _percentile(samples: np.ndarray, actual: float) -> float:
    if samples.size == 0:
        raise ValueError("score evaluation requires at least one world")
    below = float(np.mean(samples < actual))
    equal = float(np.mean(samples == actual))
    return below + 0.5 * equal


def _covered(samples: np.ndarray, actual: float, mass: float) -> bool:
    alpha = (1.0 - mass) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return bool(low <= actual <= high)


def _winner_brier(away: np.ndarray, home: np.ndarray, actual_away: int, actual_home: int) -> float:
    p_away = float(np.mean(away > home))
    p_home = float(np.mean(home > away))
    p_tie = float(np.mean(away == home))
    actual = np.asarray(
        [actual_away > actual_home, actual_home > actual_away, actual_away == actual_home],
        dtype=float,
    )
    probs = np.asarray([p_away, p_home, p_tie], dtype=float)
    return float(np.mean((probs - actual) ** 2))


def evaluate_score_worlds(
    away_points: np.ndarray,
    home_points: np.ndarray,
    *,
    actual_away: int,
    actual_home: int,
) -> ScoreDistributionEvaluation:
    """Evaluate one frozen game's world distribution against an observed result.

    This function never changes the simulator. It scores whether the observed game was well
    represented by the pre-result distribution, including tails and joint score geometry.
    """
    away = np.asarray(away_points, dtype=float)
    home = np.asarray(home_points, dtype=float)
    if away.ndim != 1 or home.ndim != 1 or away.size != home.size or away.size == 0:
        raise ValueError("away/home score worlds must be non-empty aligned one-dimensional arrays")

    total = away + home
    margin = away - home
    actual_total = actual_away + actual_home
    actual_margin = actual_away - actual_home
    exact = (away == actual_away) & (home == actual_home)
    nearby_3 = (np.abs(away - actual_away) <= 3.0) & (np.abs(home - actual_home) <= 3.0)
    nearby_7 = (np.abs(away - actual_away) <= 7.0) & (np.abs(home - actual_home) <= 7.0)

    return ScoreDistributionEvaluation(
        actual_away=actual_away,
        actual_home=actual_home,
        away_mean=float(away.mean()),
        home_mean=float(home.mean()),
        total_mean=float(total.mean()),
        margin_mean=float(margin.mean()),
        away_mae=abs(float(away.mean()) - actual_away),
        home_mae=abs(float(home.mean()) - actual_home),
        total_absolute_error=abs(float(total.mean()) - actual_total),
        margin_absolute_error=abs(float(margin.mean()) - actual_margin),
        actual_total_percentile=_percentile(total, actual_total),
        actual_margin_percentile=_percentile(margin, actual_margin),
        actual_away_percentile=_percentile(away, actual_away),
        actual_home_percentile=_percentile(home, actual_home),
        central_50_total_covered=_covered(total, actual_total, 0.50),
        central_80_total_covered=_covered(total, actual_total, 0.80),
        central_90_total_covered=_covered(total, actual_total, 0.90),
        central_50_margin_covered=_covered(margin, actual_margin, 0.50),
        central_80_margin_covered=_covered(margin, actual_margin, 0.80),
        central_90_margin_covered=_covered(margin, actual_margin, 0.90),
        away_win_probability=float(np.mean(away > home)),
        home_win_probability=float(np.mean(home > away)),
        tie_probability=float(np.mean(away == home)),
        winner_brier=_winner_brier(away, home, actual_away, actual_home),
        exact_score_probability=float(np.mean(exact)),
        nearby_score_probability_3=float(np.mean(nearby_3)),
        nearby_score_probability_7=float(np.mean(nearby_7)),
    )
