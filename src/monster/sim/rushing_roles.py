from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds
from monster.snapshot.player import TeamPlayerPool

# 2022-2025 REG audits show unusually stable rushing structure:
# - pooled ranked shares ≈ 58.63% / 24.63% / 11.30%
# - 2.47-2.56 core rushers per team-game
# - 8.46-9.21% incidental rushing share.
# We therefore retain the finite core + incidental architecture and give the selected core
# explicit Role A/B/C entitlement. These are structural priors, not deterministic projections.
INCIDENTAL_RUSH_SHARE = 0.0870
CORE_THIRD_RUSHER_PROBABILITY = 0.50
CORE_ROLE_ENTITLEMENT = np.array([0.62, 0.27, 0.11], dtype=float)
CORE_ROLE_CONCENTRATION = 22.0
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
    target = int(rng.choice(INCIDENTAL_COUNT_VALUES, p=INCIDENTAL_COUNT_PROBABILITIES))
    target = max(target, 1)
    return min(target, incidental_attempts, peripheral_count)


def _sample_core_roles(
    rng: np.random.Generator,
    eligible: np.ndarray,
    role_strength: np.ndarray,
    role_uncertainty: np.ndarray,
    count: int,
) -> np.ndarray:
    """Sample a persistent latent A/B/C ordering from evidence plus role uncertainty.

    Role identity and role volume are distinct uncertainties. Historical/current role strength
    supplies the center of each player's latent rank; player-specific role uncertainty supplies
    only the amount of Sunday-to-Sunday rank noise. Strongly separated roles therefore remain
    stable across worlds, while genuinely ambiguous backfields can swap A/B/C identities.
    """
    if count <= 0 or len(eligible) == 0:
        return np.empty(0, dtype=int)
    count = min(count, len(eligible))
    strength = np.clip(role_strength[eligible].astype(float), 1e-9, None)
    uncertainty = np.clip(role_uncertainty[eligible].astype(float), 0.025, 0.60)
    latent_score = np.log(strength) + rng.normal(0.0, uncertainty)
    order = np.argsort(latent_score)[::-1]
    return eligible[order[:count]]


def _core_role_probabilities(
    rng: np.random.Generator,
    ordered_candidates: np.ndarray,
) -> np.ndarray:
    """Sample carry entitlement for already ordered latent Role A/B/C occupants."""
    center = CORE_ROLE_ENTITLEMENT[: len(ordered_candidates)].astype(float)
    center /= center.sum()
    concentration = max(CORE_ROLE_CONCENTRATION, float(len(ordered_candidates) * 3))
    return rng.dirichlet(np.clip(center * concentration, 0.25, None))


def _redistribute_team_rushing(
    rng: np.random.Generator,
    side: TeamAllocationWorlds,
    pool: TeamPlayerPool,
) -> None:
    """Allocate finite team carries through stable Role A/B/C + incidental football states."""
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
    role_uncertainty = np.array([p.role_uncertainty for p in players], dtype=float)
    effectiveness = np.array([p.effectiveness_if_active for p in players], dtype=float)

    # Role strength defines the center of latent A/B/C identity. Uncertainty governs how often
    # nearby players can exchange ranks; it does not directly change the carry budget.
    role_strength = (
        np.sqrt(np.clip(base, 1e-9, None))
        * np.clip(role_probability, 0.01, 1.0)
        * np.clip(effectiveness, 0.25, 1.25)
    )
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

        incidental_n = int(rng.binomial(total, INCIDENTAL_RUSH_SHARE))
        incidental_n = min(incidental_n, max(total - 1, 0), 5)
        core_n = total - incidental_n

        core_count = _sample_core_count(rng, core_n, len(eligible))
        core_candidates = _sample_core_roles(
            rng,
            eligible,
            role_strength,
            role_uncertainty,
            core_count,
        )
        if len(core_candidates) == 0:
            core_candidates = np.array([int(eligible[np.argmax(base[eligible])])], dtype=int)

        core_p = _core_role_probabilities(rng, core_candidates)

        # Every selected core role receives at least one carry if the game has enough core work.
        guaranteed = min(core_n, len(core_candidates))
        attempts_out[w, core_candidates[:guaranteed]] += 1
        remaining_core = core_n - guaranteed
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
    """Apply the market-blind finite rushing-role hierarchy in-place."""
    rng = np.random.default_rng(seed)
    _redistribute_team_rushing(rng, worlds.away, away_pool)
    _redistribute_team_rushing(rng, worlds.home, home_pool)
    return worlds