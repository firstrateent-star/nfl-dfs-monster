from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool

# 2022-2025 REG audits show a remarkably stable realized target tree.
# Rank shares are approximately 29.5 / 20.8 / 15.4 / 11.6 / 8.6 / 6.0 percent,
# leaving ~8 percent for the remaining receiving tail. The 3-5 target middle tier
# accounts for roughly 2.5-2.7 players and 32-36 percent of team targets per game.
# These are structural priors for one-game anatomy, not player projections.
RANK_SHARE_PRIOR = np.array([0.295, 0.208, 0.154, 0.116, 0.086, 0.060], dtype=float)
RANK_SHARE_CONCENTRATION = 90.0
TAIL_SHARE = float(1.0 - RANK_SHARE_PRIOR.sum())


def _gamma_sum(
    rng: np.random.Generator,
    counts: np.ndarray,
    mean_per_event: float,
    shape_per_event: float = 2.2,
) -> np.ndarray:
    counts_float = counts.astype(float)
    shape = np.maximum(counts_float * shape_per_event, 1e-6)
    scale = max(mean_per_event, 0.1) / shape_per_event
    values = rng.gamma(shape, scale)
    values[counts == 0] = 0.0
    return values.astype(np.float32)


def _allocate_counts(
    rng: np.random.Generator,
    counts: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    worlds, n_players = shares.shape
    out = np.zeros((worlds, n_players), dtype=np.int16)
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


def _world_target_shares(
    rng: np.random.Generator,
    pool: TeamPlayerPool,
    worlds: int,
) -> np.ndarray:
    players = pool.players
    n_players = len(players)
    base = np.array([max(p.target_share, 0.0) for p in players], dtype=float)
    positive = base > 0.0
    if not positive.any():
        positive[:] = True
        base[:] = 1.0
    base /= base.sum()

    active_probability = np.array([p.active_probability for p in players], dtype=float)
    effectiveness = np.array([p.effectiveness_if_active for p in players], dtype=float)
    uncertainty = np.array([p.role_uncertainty for p in players], dtype=float)
    role_strength = np.sqrt(np.clip(base, 1e-9, None)) * np.clip(effectiveness, 0.25, 1.25)

    shares = np.zeros((worlds, n_players), dtype=float)
    for w in range(worlds):
        active = rng.random(n_players) < np.clip(active_probability, 0.0, 1.0)
        eligible = np.flatnonzero(active & positive)
        if len(eligible) == 0:
            eligible = np.array([int(np.argmax(base))], dtype=int)

        latent = np.log(np.clip(role_strength[eligible], 1e-9, None)) + rng.normal(
            0.0, np.clip(uncertainty[eligible], 0.025, 0.60)
        )
        ordered = eligible[np.argsort(latent)[::-1]]
        top_n = min(len(ordered), len(RANK_SHARE_PRIOR))

        center = np.zeros(len(ordered), dtype=float)
        center[:top_n] = RANK_SHARE_PRIOR[:top_n]
        if len(ordered) <= len(RANK_SHARE_PRIOR):
            center[:top_n] /= center[:top_n].sum()
        else:
            tail = ordered[top_n:]
            tail_weight = np.clip(base[tail], 1e-9, None)
            tail_weight /= tail_weight.sum()
            center[top_n:] = TAIL_SHARE * tail_weight

        sampled = rng.dirichlet(np.clip(center * RANK_SHARE_CONCENTRATION, 0.15, None))
        shares[w, ordered] = sampled

    return shares


def _redistribute_team_receiving(
    rng: np.random.Generator,
    side: TeamAllocationWorlds,
    pool: TeamPlayerPool,
) -> None:
    players = pool.players
    if not players:
        return

    worlds = len(side.team_targets)
    shares = _world_target_shares(rng, pool, worlds)
    targets = _allocate_counts(rng, side.team_targets.astype(int), shares)

    rec_td_weight = np.array(
        [max(p.receiving_td_share, p.red_zone_target_share, 0.01) for p in players],
        dtype=float,
    )
    rec_td_out = np.zeros_like(targets, dtype=np.int16)
    for w in range(worlds):
        td_total = int(side.passing_tds[w])
        if td_total <= 0:
            continue
        weight = targets[w].astype(float) * rec_td_weight
        if weight.sum() <= 0:
            weight = shares[w].copy()
        weight /= weight.sum()
        rec_td_out[w] = rng.multinomial(td_total, weight).astype(np.int16)

    pass_yard_multiplier = np.clip(1.0 - 0.32 * (side.pass_disruption - 1.0), 0.90, 1.10)
    total_receptions = np.zeros(worlds, dtype=np.int16)
    total_receiving_yards = np.zeros(worlds, dtype=np.float32)

    for idx, player in enumerate(players):
        stats = side.player_stats[player.player_id]
        catch_probability = np.clip(
            player.catch_rate
            * player.catchpoint_modifier
            * player.effectiveness_if_active
            * (1.0 - 0.12 * (side.pass_disruption - 1.0)),
            0.25,
            0.93,
        )
        receptions = rng.binomial(targets[:, idx], catch_probability).astype(np.int16)
        yards = _gamma_sum(
            rng,
            receptions,
            player.yards_per_reception * player.explosive_modifier,
        )
        yards = (yards * pass_yard_multiplier).astype(np.float32)

        stats["targets"] = targets[:, idx].astype(np.float32)
        stats["receptions"] = receptions.astype(np.float32)
        stats["receiving_yards"] = yards
        stats["receiving_tds"] = rec_td_out[:, idx].astype(np.float32)
        total_receptions += receptions
        total_receiving_yards += yards

    quarterback_indices = [idx for idx, player in enumerate(players) if player.position == "QB"]
    if quarterback_indices:
        attempt_matrix = np.column_stack(
            [side.player_stats[players[idx].player_id]["pass_attempts"] for idx in quarterback_indices]
        ).astype(float)
        attempt_sum = attempt_matrix.sum(axis=1)
        qb_shares = np.divide(
            attempt_matrix,
            np.maximum(attempt_sum[:, None], 1.0),
            out=np.zeros_like(attempt_matrix),
            where=attempt_sum[:, None] > 0,
        )
        empty = qb_shares.sum(axis=1) <= 0
        if empty.any():
            qb_shares[empty, 0] = 1.0

        qb_completions = _allocate_counts(rng, total_receptions.astype(int), qb_shares)
        qb_passing_tds = _allocate_counts(rng, side.passing_tds.astype(int), qb_shares)
        for local_idx, player_idx in enumerate(quarterback_indices):
            stats = side.player_stats[players[player_idx].player_id]
            stats["completions"] = qb_completions[:, local_idx].astype(np.float32)
            stats["passing_yards"] = (
                total_receiving_yards * qb_shares[:, local_idx]
            ).astype(np.float32)
            stats["passing_tds"] = qb_passing_tds[:, local_idx].astype(np.float32)

    assert np.array_equal(targets.sum(axis=1), side.team_targets.astype(int))
    assert np.array_equal(rec_td_out.sum(axis=1), side.passing_tds.astype(int))


def refine_game_receiving_roles(
    worlds: GameAllocationWorlds,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    seed: int,
) -> GameAllocationWorlds:
    """Apply the market-blind finite receiving hierarchy in-place."""
    rng = np.random.default_rng(seed)
    _redistribute_team_receiving(rng, worlds.away, away_pool)
    _redistribute_team_receiving(rng, worlds.home, home_pool)
    return worlds
