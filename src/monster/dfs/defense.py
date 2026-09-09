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


def score_defense_partial_worlds(*, opponent_points: np.ndarray, opponent_turnovers: np.ndarray) -> np.ndarray:
    points = np.asarray(opponent_points)
    turnovers = np.asarray(opponent_turnovers)
    if points.shape != turnovers.shape:
        raise ValueError("Defense inputs must share one correlated world shape")
    return points_allowed_score(points) + 2.0 * turnovers.astype(np.float32)


def score_defense_worlds(
    *,
    opponent_points: np.ndarray,
    opponent_turnovers: np.ndarray,
    sacks: np.ndarray,
    defensive_touchdowns: np.ndarray,
    special_teams_touchdowns: np.ndarray,
    safeties: np.ndarray,
) -> np.ndarray:
    """Complete currently modeled FanDuel D/ST score in one correlated Sunday world."""
    arrays = [
        np.asarray(opponent_points), np.asarray(opponent_turnovers), np.asarray(sacks),
        np.asarray(defensive_touchdowns), np.asarray(special_teams_touchdowns), np.asarray(safeties),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Complete D/ST inputs must share one correlated world shape")
    return (
        points_allowed_score(arrays[0])
        + 2.0 * arrays[1].astype(np.float32)
        + arrays[2].astype(np.float32)
        + 6.0 * arrays[3].astype(np.float32)
        + 6.0 * arrays[4].astype(np.float32)
        + 2.0 * arrays[5].astype(np.float32)
    )


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
