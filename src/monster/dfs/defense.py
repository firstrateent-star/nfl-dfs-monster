from __future__ import annotations

import numpy as np


def points_allowed_score(points_allowed: np.ndarray) -> np.ndarray:
    """FanDuel D/ST points-allowed component for regulation game worlds."""
    points = np.asarray(points_allowed)
    out = np.empty(points.shape, dtype=np.float32)
    out[points == 0] = 10.0
    out[(points >= 1) & (points <= 6)] = 7.0
    out[(points >= 7) & (points <= 13)] = 4.0
    out[(points >= 14) & (points <= 20)] = 1.0
    out[(points >= 21) & (points <= 27)] = 0.0
    out[(points >= 28) & (points <= 34)] = -1.0
    out[points >= 35] = -4.0
    return out


def score_defense_partial_worlds(
    *,
    opponent_points: np.ndarray,
    opponent_turnovers: np.ndarray,
) -> np.ndarray:
    """Score only D/ST components already represented by certified game worlds.

    Each opponent turnover contributes the standard +2 takeaway points. Sacks and defensive/
    special-teams touchdowns are deliberately excluded until those events exist explicitly in the
    football world. This prevents the optimizer from receiving invented independent defense tails.
    """
    points = np.asarray(opponent_points)
    turnovers = np.asarray(opponent_turnovers)
    if points.shape != turnovers.shape:
        raise ValueError("Defense inputs must share one correlated world shape")
    return points_allowed_score(points) + 2.0 * turnovers.astype(np.float32)


def require_complete_defense_authority(
    *, sacks_modeled: bool, defensive_scores_modeled: bool, special_teams_scores_modeled: bool
) -> None:
    """Prevent partial D/ST worlds from silently entering authoritative lineup optimization."""
    missing = []
    if not sacks_modeled:
        missing.append("sacks")
    if not defensive_scores_modeled:
        missing.append("defensive touchdowns/safeties")
    if not special_teams_scores_modeled:
        missing.append("special-teams touchdowns/blocks")
    if missing:
        raise RuntimeError("FanDuel D/ST authority incomplete: " + ", ".join(missing))
