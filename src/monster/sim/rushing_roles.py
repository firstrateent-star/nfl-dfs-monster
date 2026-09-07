from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool

# 2025 REG diagnostic over 544 team-games:
# - 2.48 core rushers (>=3 carries), median 2, p90 3
# - 1.55 incidental rushers (1-2 carries), median 1, p90 3
# - 8.69% of team carries went to <=2-carry rushers.
# These are candidate structural priors until multi-season OOS validation promotes them.
INCIDENTAL_RUSH_SHARE = 0.0869
CORE_THIRD_RUSHER_PROBABILITY = 0.48
CORE_HIERARCHY_POWER = 1.35
INCIDENTAL_COUNT_VALUES = np.array([0, 1, 2, 3], dtype=np.int8)
INCIDENTAL_COUNT_PROBABILITIES = np.array([0.15, 0.35, 0.30, 0.20], dtype=float)


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


def _sample_core_count(rng: np.random.Generator, total_carries: int, eligible_count: int) -> int:
    """Sample a finite 2-3 player core rushing set from the observed 2025 structure."""
    if total_carries <= 0 or eligible_count <= 0:
        return 0
    if total_carries == 1 or eligible_count == 1:
        return 1
    target = 2 + int(rng.random() < CORE_THIRD_RUSHER_PROBABILITY)
    return min(target, eligible_count, total_carries)


def _sample_incidental_count(
    rng: np.random.Generator,
    incidental_attempts: int,
    peripheral_count: int,
) -> int:
    if incidental_attempts <= 0 or peripheral_count <= 0:
        return 0
    target = int(
        rng.choice(INCIDENTAL_COUNT_VALUES, p=INCIDENTAL_COUNT_PROBABILITIES)
    )
    # If there is an incidental attempt budget, at least one peripheral player receives it.
    target = max(target, 1)
    return min(target, incidental_attempts, peripheral_count)


def _redistribute_team_rushing(
    rng: np.random.Generator,
    side: TeamAllocationWorlds,
    pool: TeamPlayerPool,
) -> None:
    """Allocate one finite carry supply through explicit core + incidental role sets.

    Core participation is sampled as a finite roster state rather than independent Bernoulli
    eligibility across every skill player. Incidental carries then create the empirically observed
    1-2 carry tail without turning peripheral players into committee members. Team carries and
    rushing TDs are conserved exactly in every world.
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

    active_probability = np.array([p.active_probability for p in players], dtype=float)
    role_probability = np.array([p.rush_role_probability for p in players], dtype=float)
    effectiveness = np.array([p.effectiveness_if_active for p in players], dtype=float)

    # Selection answers "who belongs in the core tree?"; allocation answers "how much of that
    # tree belongs to each selected player?". Keeping these separate prevents role probability
    # from acting as an accidental second carry-share multiplier.
    selection_weight = (
        np.sqrt(np.clip(base, 1e-9, None))
        * np.clip(role_probability, 0.01, 1.0)
        * np.clip(effectiveness, 0.25, 1.25)
    )
    core_weight = np.power(np.clip(base, 1e-9, None), CORE_HIERARCHY_POWER)
    core_weight *= np.clip(effectiveness, 0.25, 1.25)
    peripheral_weight = (
        np.sqrt(np.clip(base, 1e-9, None))
        * (0.10 + 0.90 * (1.0 - role_probability))
        * np.clip(effectiveness, 0.25, 1.25)
    )

    attempts_out = np.zeros((worlds, n_players), dtype=np.int16)
    rush_td_out = np.zeros((worlds, n_players), dtype=np.int16)
    team_rushes = side.team_rush_attempts.astype(int)
    team_rush_tds = side.rushing_tds.astype(int)

    for w in range(worlds):
        total = int(team_rushes[w])
        if total <= 0:
            continue

        active = rng.random(n_players) < np.clip(active_probability, 0.0, 1.0)
        eligible = np.flatnonzero(active & (base > 0))
        if len(eligible) == 0:
            eligible = np.array([int(np.argmax(base))], dtype=int)

        # One finite team carry supply; incidental work is carved out, never added.
        incidental_n = int(rng.binomial(total, INCIDENTAL_RUSH_SHARE))
        incidental_n = min(incidental_n, max(total - 1, 0), 5)
        core_n = total - incidental_n

        core_count = _sample_core_count(rng, core_n, len(eligible))
        core_candidates = _weighted_choice_without_replacement(
            rng, eligible, selection_weight, core_count
        )
        if len(core_candidates) == 0:
            core_candidates = np.array([int(eligible[np.argmax(base[eligible])])], dtype=int)

        # Give every selected core rusher one carry first. Remaining core work follows the
        # current player hierarchy. This makes "core participant" a meaningful game state.
        guaranteed = min(core_n, len(core_candidates))
        attempts_out[w, core_candidates[:guaranteed]] += 1
        remaining_core = core_n - guaranteed
        core_p = core_weight[core_candidates].astype(float)
        if core_p.sum() <= 0:
            core_p = np.ones(len(core_candidates), dtype=float)
        core_p /= core_p.sum()
        if remaining_core > 0:
            attempts_out[w, core_candidates] += rng.multinomial(
                remaining_core, core_p
            ).astype(np.int16)

        peripheral = np.setdiff1d(eligible, core_candidates, assume_unique=False)
        incidental_count = _sample_incidental_count(rng, incidental_n, len(peripheral))
        if incidental_count > 0:
            chosen = _weighted_choice_without_replacement(
                rng, peripheral, peripheral_weight, incidental_count
            )
            # One attempt to each peripheral participant first; extra incidental carries can
            # become a second attempt, reproducing the observed 1-2 carry tail.
            attempts_out[w, chosen] += 1
            remaining_incidental = incidental_n - len(chosen)
            if remaining_incidental > 0:
                chosen_weight = peripheral_weight[chosen].astype(float)
                if chosen_weight.sum() <= 0:
                    chosen_weight = np.ones(len(chosen), dtype=float)
                chosen_weight /= chosen_weight.sum()
                attempts_out[w, chosen] += rng.multinomial(
                    remaining_incidental, chosen_weight
                ).astype(np.int16)
        elif incidental_n > 0:
            # No available peripheral role: return the tiny budget to the already-selected core.
            attempts_out[w, core_candidates] += rng.multinomial(
                incidental_n, core_p
            ).astype(np.int16)

        td_total = int(team_rush_tds[w])
        if td_total > 0:
            td_weight = attempts_out[w].astype(float) * np.array(
                [max(p.red_zone_rush_share, p.rushing_td_share, 0.01) for p in players],
                dtype=float,
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
    """Apply the candidate finite core + incidental rushing mechanism in-place."""
    rng = np.random.default_rng(seed)
    _redistribute_team_rushing(rng, worlds.away, away_pool)
    _redistribute_team_rushing(rng, worlds.home, home_pool)
    return worlds
