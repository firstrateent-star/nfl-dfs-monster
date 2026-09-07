from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from monster.snapshot.model import TeamState


def _num(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return float(value)


def team_state_from_policy_row(
    row: dict[str, Any],
    *,
    opponent_id: str,
    prior_uncertainty: float = 0.12,
    coaching_entropy: float = 0.10,
) -> TeamState:
    """Convert one market-blind historical policy prior into simulation state.

    Historical behavior is a prior, never a frozen 2026 forecast. Missing channels fall
    back to structural league-neutral defaults, while `prior_uncertainty` explicitly
    represents year-over-year epistemic uncertainty until current coaching/personnel
    evidence narrows it.
    """
    team_id = str(row["team_id"])
    seconds_per_play = _num(row, "neutral_seconds_per_play", 28.0)
    pace_factor = float(np.clip(28.0 / max(seconds_per_play, 15.0), 0.82, 1.18))

    return TeamState(
        team_id=team_id,
        opponent_id=opponent_id,
        neutral_pass_rate=float(np.clip(_num(row, "neutral_pass_rate", 0.56), 0.34, 0.72)),
        pace_factor=pace_factor,
        drives_per_game=float(np.clip(_num(row, "drives_per_game", 10.5), 7.5, 14.0)),
        td_drive_rate=float(np.clip(_num(row, "td_drive_rate", 0.22), 0.08, 0.42)),
        fg_drive_rate=float(np.clip(_num(row, "fg_drive_rate", 0.14), 0.04, 0.28)),
        turnover_drive_rate=float(np.clip(_num(row, "turnover_drive_rate", 0.11), 0.04, 0.24)),
        red_zone_td_rate=float(np.clip(_num(row, "red_zone_td_rate", 0.55), 0.28, 0.80)),
        offensive_epa_per_play=float(np.clip(_num(row, "offensive_epa_per_play", 0.0), -0.25, 0.30)),
        offensive_success_rate=float(np.clip(_num(row, "offensive_success_rate", 0.44), 0.28, 0.62)),
        offensive_explosive_rate=float(np.clip(_num(row, "offensive_explosive_rate", 0.10), 0.03, 0.24)),
        defensive_td_drive_rate_allowed=float(
            np.clip(_num(row, "defensive_td_drive_rate_allowed", 0.22), 0.08, 0.42)
        ),
        defensive_fg_drive_rate_allowed=float(
            np.clip(_num(row, "defensive_fg_drive_rate_allowed", 0.14), 0.04, 0.28)
        ),
        defensive_takeaway_drive_rate=float(
            np.clip(_num(row, "defensive_takeaway_drive_rate", 0.11), 0.04, 0.24)
        ),
        defensive_epa_allowed_per_play=float(
            np.clip(_num(row, "defensive_epa_allowed_per_play", 0.0), -0.25, 0.30)
        ),
        defensive_explosive_rate_allowed=float(
            np.clip(_num(row, "defensive_explosive_rate_allowed", 0.10), 0.03, 0.24)
        ),
        defensive_sack_rate=float(np.clip(_num(row, "defensive_sack_rate", 0.07), 0.02, 0.16)),
        defensive_qb_hit_rate=float(
            np.clip(_num(row, "defensive_qb_hit_rate", 0.18), 0.06, 0.38)
        ),
        coaching_entropy=float(np.clip(coaching_entropy, 0.02, 0.35)),
        uncertainty=float(np.clip(prior_uncertainty, 0.04, 0.35)),
    )


def compile_team_state_map(
    policy: pl.DataFrame,
    opponents: dict[str, str],
    *,
    prior_uncertainty: float = 0.12,
) -> dict[str, TeamState]:
    """Compile all teams for a slate/game set from one historical policy artifact."""
    if "team_id" not in policy.columns:
        raise ValueError("Policy artifact requires team_id")
    rows = {str(row["team_id"]): row for row in policy.to_dicts()}
    states: dict[str, TeamState] = {}
    for team_id, opponent_id in opponents.items():
        row = rows.get(team_id, {"team_id": team_id})
        states[team_id] = team_state_from_policy_row(
            row,
            opponent_id=opponent_id,
            prior_uncertainty=prior_uncertainty,
        )
    return states
