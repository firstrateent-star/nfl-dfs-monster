from collections.abc import Mapping

import numpy as np

# DraftKings Classic NFL offensive scoring.  This module is intentionally downstream
# of football simulation: salary, ownership and lineup construction never enter here.
DRAFTKINGS_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -1.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receptions": 1.0,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "fumbles_lost": -1.0,
}


def score_offensive_player_worlds(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    """Translate correlated football-world stats into DraftKings fantasy points.

    Bonuses are calculated inside each simulated world, preserving the nonlinear value
    of 300-yard passing and 100-yard rushing/receiving outcomes. Missing football state
    is rejected rather than silently treated as zero.
    """
    required = tuple(DRAFTKINGS_SCORING)
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Incomplete DraftKings scoring state; missing: {', '.join(missing)}")

    first = np.asarray(stats[required[0]], dtype=float)
    points = np.zeros_like(first, dtype=float)
    arrays: dict[str, np.ndarray] = {}
    for stat, weight in DRAFTKINGS_SCORING.items():
        values = np.asarray(stats[stat], dtype=float)
        if values.shape != points.shape:
            raise ValueError(f"DraftKings stat shape mismatch for {stat}")
        arrays[stat] = values
        points += values * weight

    points += 3.0 * (arrays["passing_yards"] >= 300.0)
    points += 3.0 * (arrays["rushing_yards"] >= 100.0)
    points += 3.0 * (arrays["receiving_yards"] >= 100.0)
    return points
