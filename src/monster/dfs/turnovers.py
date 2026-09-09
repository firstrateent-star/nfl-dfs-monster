from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.allocation import TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class TurnoverAttribution:
    interceptions: dict[str, np.ndarray]
    fumbles_lost: dict[str, np.ndarray]
    team_interceptions: np.ndarray
    team_fumbles_lost: np.ndarray


def _allocate_counts(
    rng: np.random.Generator, counts: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Allocate integer events while conserving every world total exactly."""
    worlds, n_players = weights.shape
    out = np.zeros((worlds, n_players), dtype=np.int16)
    row_sum = weights.sum(axis=1)
    empty = row_sum <= 0
    if empty.any():
        weights = weights.copy()
        weights[empty, 0] = 1.0
        row_sum = weights.sum(axis=1)
    shares = weights / row_sum[:, None]
    remaining = counts.astype(np.int64).copy()
    remaining_share = np.ones(worlds, dtype=float)
    for idx in range(n_players - 1):
        p = np.divide(
            shares[:, idx],
            np.maximum(remaining_share, 1e-12),
            out=np.zeros(worlds, dtype=float),
            where=remaining_share > 0,
        )
        draw = rng.binomial(remaining, np.clip(p, 0.0, 1.0))
        out[:, idx] = draw.astype(np.int16)
        remaining -= draw
        remaining_share -= shares[:, idx]
    out[:, -1] = remaining.astype(np.int16)
    return out


def attribute_team_turnovers(
    team_turnovers: np.ndarray,
    allocation: TeamAllocationWorlds,
    pool: TeamPlayerPool,
    *,
    interception_fraction: float,
    seed: int,
) -> TurnoverAttribution:
    """Decompose an already-simulated team turnover reservoir into DFS penalties.

    This layer is downstream-only: it never changes team turnover counts or football scores.
    Interception-vs-fumble mix must be supplied from football-only historical calibration.
    Interceptions are allocated only to QBs with pass attempts in that world. Fumbles lost are
    allocated to offensive ball handlers in proportion to simulated touches (attempts + catches),
    with QB dropback exposure added so strip-sack risk is not structurally omitted.
    """
    turnovers = np.asarray(team_turnovers, dtype=np.int16)
    if turnovers.ndim != 1:
        raise ValueError("team_turnovers must be one-dimensional")
    fraction = float(interception_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("interception_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    team_interceptions = rng.binomial(turnovers.astype(np.int64), fraction).astype(np.int16)
    team_fumbles = (turnovers - team_interceptions).astype(np.int16)

    players = pool.players
    worlds = len(turnovers)
    qb_weights = np.zeros((worlds, len(players)), dtype=float)
    fumble_weights = np.zeros((worlds, len(players)), dtype=float)
    for idx, player in enumerate(players):
        stats = allocation.player_stats[player.player_id]
        pass_attempts = np.asarray(stats["pass_attempts"], dtype=float)
        rush_attempts = np.asarray(stats["rush_attempts"], dtype=float)
        receptions = np.asarray(stats["receptions"], dtype=float)
        if player.position == "QB":
            qb_weights[:, idx] = pass_attempts
            # Dropbacks expose QBs to both scrambles and strip sacks. Pass attempts are a
            # conservative proxy because sacks are team-level in the frozen allocation state.
            fumble_weights[:, idx] = rush_attempts + receptions + 0.20 * pass_attempts
        else:
            fumble_weights[:, idx] = rush_attempts + receptions

    # If a turnover exists in a world with no simulated QB attempt, assign the interception to
    # the highest-current QB role rather than leaking it to a non-QB.
    qb_indices = [i for i, player in enumerate(players) if player.position == "QB"]
    if team_interceptions.sum() and not qb_indices:
        raise ValueError(f"Team {pool.team_id} has interceptions but no QB in player pool")
    if qb_indices:
        qb_empty = qb_weights.sum(axis=1) <= 0
        fallback_qb = max(qb_indices, key=lambda i: players[i].qb_pass_share)
        qb_weights[qb_empty, fallback_qb] = 1.0

    # A turnover world can theoretically have no skill-player touch because of contingent role
    # states. Use total simulated offensive involvement as a deterministic fallback.
    fumble_empty = fumble_weights.sum(axis=1) <= 0
    if fumble_empty.any():
        fallback = max(range(len(players)), key=lambda i: players[i].active_probability)
        fumble_weights[fumble_empty, fallback] = 1.0

    int_matrix = _allocate_counts(rng, team_interceptions, qb_weights)
    fum_matrix = _allocate_counts(rng, team_fumbles, fumble_weights)

    return TurnoverAttribution(
        interceptions={p.player_id: int_matrix[:, i].astype(np.float32) for i, p in enumerate(players)},
        fumbles_lost={p.player_id: fum_matrix[:, i].astype(np.float32) for i, p in enumerate(players)},
        team_interceptions=team_interceptions,
        team_fumbles_lost=team_fumbles,
    )


def attach_turnover_stats(
    allocation: TeamAllocationWorlds, attribution: TurnoverAttribution
) -> None:
    """Attach downstream turnover penalties to existing player-world stat dictionaries."""
    for player_id, stats in allocation.player_stats.items():
        stats["interceptions"] = attribution.interceptions[player_id]
        stats["fumbles_lost"] = attribution.fumbles_lost[player_id]
