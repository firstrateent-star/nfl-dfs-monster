from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool

# 2025 diagnostic: players finishing with <=2 carries accounted for 8.69% of team rushes.
# Keep this as a structural candidate until multi-season OOS validation promotes it.
INCIDENTAL_RUSH_SHARE = 0.0869
CORE_HIERARCHY_POWER = 1.35


def _gamma_sum(
    rng: np.random.Generator,
    counts: np.ndarray,
    mean_per_event: float,
    shape_per_event: float = 2.8,
) -> np.ndarray:
    counts_float = counts.astype(float)
    shape = np.maximum(counts_float * shape_per_event, 1e-6)
    scale = max(mean_per_event, 0.1) / shape_per_event
    values = rng.gamma(shape, scale)
    values[counts == 0] = 0.0
    return values.astype(np.float32)


def _weighted_choice_without_replacement(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or len(candidates) == 0:
        return np.empty(0, dtype=int)
    count = min(count, len(candidates))
    local = np.clip(weights[candidates].astype(float), 0.0, None)
    if local.sum() <= 0:
        local = np.ones(len(candidates), dtype=float)
    local /= local.sum()
    return rng.choice(candidates, size=count, replace=False, p=local)


def _redistribute_team_rushing(
    rng: np.random.Generator,
    side: TeamAllocationWorlds,
    pool: TeamPlayerPool,
) -> None:
    """Split finite team carries into core work and tiny incidental player-game work.

    The team rush total is conserved exactly. A core carry tree receives roughly 91% of attempts;
    peripheral players can enter through an incidental budget designed to create 1-2 carry game
    appearances without stealing a full committee share. This is a candidate mechanism, not an
    asserted NFL law, and remains subject to multi-season OOS validation.
    """
    players = pool.players
    if not players:
        return

    worlds = len(side.team_rush_attempts)
    n_players = len(players)
    base = np.array([max(p.rush_share, 0.0) for p in players], dtype=float)
    if base.sum() <= 0:
        return
    base /= base.sum()
    core_weight = np.power(np.clip(base, 1e-9, None), CORE_HIERARCHY_POWER)
    core_weight /= core_weight.sum()

    active_probability = np.array([p.active_probability for p in players], dtype=float)
    role_probability = np.array([p.rush_role_probability for p in players], dtype=float)
    effectiveness = np.array([p.effectiveness_if_active for p in players], dtype=float)
    peripheral_weight = np.sqrt(np.clip(base, 1e-9, None)) * (0.15 + 0.85 * (1.0 - role_probability))

    attempts_out = np.zeros((worlds, n_players), dtype=np.int16)
    rush_td_out = np.zeros((worlds, n_players), dtype=np.int16)
    team_rushes = side.team_rush_attempts.astype(int)
    team_rush_tds = side.rushing_tds.astype(int)

    for w in range(worlds):
        total = int(team_rushes[w])
        if total <= 0:
            continue

        active = rng.random(n_players) < np.clip(active_probability, 0.0, 1.0)
        core_active = active & (rng.random(n_players) < np.clip(role_probability, 0.0, 1.0)) & (base > 0)
        if not core_active.any():
            fallback = int(np.argmax(base * active)) if active.any() else int(np.argmax(base))
            core_active[fallback] = True

        # One finite team carry supply; the incidental budget is carved out of it, never added.
        incidental_n = int(rng.binomial(total, INCIDENTAL_RUSH_SHARE))
        incidental_n = min(incidental_n, max(total - 1, 0), 5)
        core_n = total - incidental_n

        core_candidates = np.flatnonzero(core_active)
        core_p = core_weight[core_candidates] * np.clip(effectiveness[core_candidates], 0.25, 1.25)
        if core_p.sum() <= 0:
            core_p = np.ones(len(core_candidates), dtype=float)
        core_p /= core_p.sum()
        attempts_out[w, core_candidates] += rng.multinomial(core_n, core_p).astype(np.int16)

        peripheral = np.flatnonzero(active & (~core_active) & (base > 0))
        if incidental_n > 0 and len(peripheral):
            # Historical rank-4+ rushers are overwhelmingly 1-2 carry appearances. Spread the
            # incidental budget across multiple eligible players before awarding second attempts.
            participant_count = 1
            if incidental_n >= 2 and rng.random() < 0.58:
                participant_count += 1
            if incidental_n >= 3 and rng.random() < 0.32:
                participant_count += 1
            participant_count = min(participant_count, incidental_n, len(peripheral), 3)
            chosen = _weighted_choice_without_replacement(
                rng, peripheral, peripheral_weight, participant_count
            )
            attempts_out[w, chosen] += 1
            remaining = incidental_n - participant_count
            if remaining > 0:
                chosen_weight = peripheral_weight[chosen]
                if chosen_weight.sum() <= 0:
                    chosen_weight = np.ones(len(chosen), dtype=float)
                chosen_weight = chosen_weight / chosen_weight.sum()
                attempts_out[w, chosen] += rng.multinomial(remaining, chosen_weight).astype(np.int16)
        elif incidental_n > 0:
            # If nobody outside the core tree is active, return those carries to the core rather
            # than inventing a ghost rusher.
            attempts_out[w, core_candidates] += rng.multinomial(incidental_n, core_p).astype(np.int16)

        td_total = int(team_rush_tds[w])
        if td_total > 0:
            td_weight = attempts_out[w].astype(float) * np.array(
                [max(p.red_zone_rush_share, p.rushing_td_share, 0.01) for p in players], dtype=float
            )
            if td_weight.sum() <= 0:
                td_weight = attempts_out[w].astype(float)
            td_weight /= td_weight.sum()
            rush_td_out[w] = rng.multinomial(td_total, td_weight).astype(np.int16)

    # Recompute player rushing production from the newly conserved attempt worlds.
    for idx, player in enumerate(players):
        stats = side.player_stats[player.player_id]
        counts = attempts_out[:, idx]
        rush_mean = (
            player.yards_per_carry
            * player.rushing_efficiency_modifier
            * player.effectiveness_if_active
        )
        yards = _gamma_sum(rng, counts, max(rush_mean, 0.5), 2.8)
        stats["rush_attempts"] = counts.astype(np.float32)
        stats["rushing_yards"] = (yards * side.run_efficiency).astype(np.float32)
        stats["rushing_tds"] = rush_td_out[:, idx].astype(np.float32)

    assert np.array_equal(attempts_out.sum(axis=1), side.team_rush_attempts.astype(int))
    assert np.array_equal(rush_td_out.sum(axis=1), side.rushing_tds.astype(int))


def refine_game_rushing_roles(
    worlds: GameAllocationWorlds,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    seed: int,
) -> GameAllocationWorlds:
    """Apply the candidate two-tier rushing role mechanism in-place and return the game worlds."""
    rng = np.random.default_rng(seed)
    _redistribute_team_rushing(rng, worlds.away, away_pool)
    _redistribute_team_rushing(rng, worlds.home, home_pool)
    return worlds
