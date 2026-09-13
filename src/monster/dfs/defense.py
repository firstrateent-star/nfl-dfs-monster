from __future__ import annotations

import numpy as np


def points_allowed_score(points_allowed: np.ndarray) -> np.ndarray:
    """FanDuel D/ST points-allowed component for correlated game worlds."""
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
    blocked_kicks: np.ndarray | None = None,
) -> np.ndarray:
    """Score the FanDuel D/ST components represented by one Monster football world.

    Turnovers are recoveries/interceptions earned by the D/ST, not turnover-on-downs. Return
    touchdowns remain separated only for auditability; FanDuel awards six points to either
    defensive or special-teams return scores. Blocked punts/field goals receive two points.
    """
    arrays = [
        np.asarray(opponent_points),
        np.asarray(opponent_turnovers),
        np.asarray(sacks),
        np.asarray(defensive_touchdowns),
        np.asarray(special_teams_touchdowns),
        np.asarray(safeties),
    ]
    blocks = np.zeros_like(arrays[0], dtype=float) if blocked_kicks is None else np.asarray(blocked_kicks)
    arrays.append(blocks)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Complete D/ST inputs must share one correlated world shape")
    return (
        points_allowed_score(arrays[0])
        + 2.0 * arrays[1].astype(np.float32)
        + arrays[2].astype(np.float32)
        + 6.0 * arrays[3].astype(np.float32)
        + 6.0 * arrays[4].astype(np.float32)
        + 2.0 * arrays[5].astype(np.float32)
        + 2.0 * arrays[6].astype(np.float32)
    )


def require_complete_defense_authority(
    *,
    sacks_modeled: bool,
    defensive_scores_modeled: bool,
    special_teams_scores_modeled: bool,
    blocked_kicks_modeled: bool = False,
) -> None:
    """Prevent an incomplete D/ST representation from entering authoritative optimization."""
    missing = []
    if not sacks_modeled:
        missing.append("sacks")
    if not defensive_scores_modeled:
        missing.append("defensive touchdowns/safeties")
    if not special_teams_scores_modeled:
        missing.append("special-teams return touchdowns")
    if not blocked_kicks_modeled:
        missing.append("blocked punts/field goals")
    if missing:
        raise RuntimeError("FanDuel D/ST authority incomplete: " + ", ".join(missing))
