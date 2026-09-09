from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool


def _capacity_weighted_allocation(
    rng: np.random.Generator,
    totals: np.ndarray,
    capacities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Allocate finite events without assigning more outcomes than source events.

    Each touchdown must belong to an event that actually exists in the same simulated world.
    This turns TD attribution into a child of receptions/carries rather than an independent
    fantasy-facing share draw.
    """
    totals = np.asarray(totals, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    if capacities.shape != weights.shape:
        raise ValueError("capacities and weights must have identical shape")
    if capacities.shape[0] != len(totals):
        raise ValueError("one capacity row is required per world")

    out = np.zeros_like(capacities, dtype=np.int16)
    for w, total in enumerate(totals):
        total = int(total)
        if total <= 0:
            continue
        remaining_capacity = capacities[w].copy()
        if int(remaining_capacity.sum()) < total:
            raise ValueError(
                "touchdown supply exceeds eligible source events; football state is inconsistent"
            )
        for _ in range(total):
            eligible = remaining_capacity > 0
            local = np.where(eligible, np.clip(weights[w], 0.0, None), 0.0)
            if local.sum() <= 0:
                local = remaining_capacity.astype(float)
            local /= local.sum()
            idx = int(rng.choice(len(local), p=local))
            out[w, idx] += 1
            remaining_capacity[idx] -= 1
    return out


def _couple_team_touchdowns(
    rng: np.random.Generator,
    side: TeamAllocationWorlds,
    pool: TeamPlayerPool,
) -> None:
    players = pool.players
    if not players:
        return

    receptions = np.column_stack(
        [side.player_stats[p.player_id]["receptions"] for p in players]
    ).astype(np.int16)
    carries = np.column_stack(
        [side.player_stats[p.player_id]["rush_attempts"] for p in players]
    ).astype(np.int16)

    rec_trait = np.array(
        [max(p.receiving_td_share, p.red_zone_target_share, 0.01) for p in players],
        dtype=float,
    )
    rush_trait = np.array(
        [max(p.rushing_td_share, p.red_zone_rush_share, 0.01) for p in players],
        dtype=float,
    )
    rec_weights = receptions.astype(float) * rec_trait[None, :]
    rush_weights = carries.astype(float) * rush_trait[None, :]

    rec_tds = _capacity_weighted_allocation(
        rng, side.passing_tds.astype(int), receptions, rec_weights
    )
    rush_tds = _capacity_weighted_allocation(
        rng, side.rushing_tds.astype(int), carries, rush_weights
    )

    for idx, player in enumerate(players):
        stats = side.player_stats[player.player_id]
        stats["receiving_tds"] = rec_tds[:, idx].astype(np.float32)
        stats["rushing_tds"] = rush_tds[:, idx].astype(np.float32)

    # Passing TD ownership remains a QB event. Reallocate from the same finite team passing-TD
    # supply using actual pass-attempt participation so receiver and QB TD totals remain identical.
    qb_indices = [idx for idx, p in enumerate(players) if p.position == "QB"]
    if qb_indices:
        attempts = np.column_stack(
            [side.player_stats[players[idx].player_id]["pass_attempts"] for idx in qb_indices]
        ).astype(float)
        qb_tds = np.zeros((len(side.passing_tds), len(qb_indices)), dtype=np.int16)
        for w, total in enumerate(side.passing_tds.astype(int)):
            if total <= 0:
                continue
            local = np.clip(attempts[w], 0.0, None)
            if local.sum() <= 0:
                local = np.ones(len(qb_indices), dtype=float)
            local /= local.sum()
            qb_tds[w] = rng.multinomial(int(total), local).astype(np.int16)
        for local_idx, player_idx in enumerate(qb_indices):
            side.player_stats[players[player_idx].player_id]["passing_tds"] = qb_tds[
                :, local_idx
            ].astype(np.float32)

    assert np.array_equal(rec_tds.sum(axis=1), side.passing_tds.astype(int))
    assert np.array_equal(rush_tds.sum(axis=1), side.rushing_tds.astype(int))
    assert np.all(rec_tds <= receptions)
    assert np.all(rush_tds <= carries)


def couple_touchdowns_to_events(
    worlds: GameAllocationWorlds,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    seed: int,
) -> GameAllocationWorlds:
    """Make final player TDs children of final receptions/carries in each football world."""
    rng = np.random.default_rng(seed)
    _couple_team_touchdowns(rng, worlds.away, away_pool)
    _couple_team_touchdowns(rng, worlds.home, home_pool)
    return worlds
