from collections.abc import Mapping

import numpy as np

FANDUEL_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -1.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receptions": 0.50,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
}


def score_offensive_player_worlds(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    """Translate football-world player stats into FanDuel points.

    This is deliberately downstream of the frozen football model. Missing turnover
    attribution is an error rather than an implicit zero: DFS scoring cannot silently
    redefine football reality or certify incomplete fantasy points.
    """
    required = tuple(FANDUEL_SCORING)
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Incomplete FanDuel scoring state; missing: {', '.join(missing)}")

    first = np.asarray(stats[required[0]], dtype=float)
    points = np.zeros_like(first, dtype=float)
    for stat, weight in FANDUEL_SCORING.items():
        values = np.asarray(stats[stat], dtype=float)
        if values.shape != points.shape:
            raise ValueError(f"FanDuel stat shape mismatch for {stat}")
        points += values * weight
    return points
