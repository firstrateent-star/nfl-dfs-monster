from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, TeamAllocationWorlds, _gamma_sum
from monster.snapshot.player import TeamPlayerPool

INCIDENTAL_RUSH_SHARE = 0.08


def _weighted_choice_without_replacement(
    rng: np.random.Generator,
    indices: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or len(indices) == 0:
        return np.array([], dtype=int)
    count = min(count, len(indices))
    p = np.clip(weights.astype(float), 0.0, None)
    if p.sum() <= 0:
        p = np.ones(len(indices), dtype=float)
    p /= p.sum()
    return rng.choice(indices, size=count, replace=False, p=p)


def _sample_core_count(rng: np.random.Generator, carries: int, eligible_count: int) -> int:
    if carries <= 0 or eligible_count <= 0:
        return 0
    if carries <= 5:
        probs = np.array([0.78, 0.20, 0.02])
    elif carries <= 12:
        probs = np.array([0.50, 0.42, 0.08])
    else:
        probs = np.array([0.25, 0.55, 0.20])
    return min(int(rng.choice(np.array([1, 2, 3]), p=probs)), eligible_count, carries)


def _sample_incidental_count(
    rng: np.random.Generator,
    carries: int,
    eligible_count: int,
) -> int:
    if carries <= 0 or eligible_count <= 0:
        return 0
    if carries == 1:
        return 1
    return min(int(rng.choice(np.array([1, 2, 3]), p=np.array([0.72, 0.23, 0.05]))), eligible_count, carries)


def _sample_core_roles(
    rng: np.random.Generator,
    eligible: np.ndarray,
    role_strength: np.ndarray,
    role_uncertainty: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or len(eligible) == 0:
        return np.array([], dtype=int)
    sigma = 0.08 + 0.55 * np.clip(role_uncertainty[eligible], 0.0, 1.0)
    latent = np.log(np.clip(role_strength[eligible], 1e-9, None)) + rng.normal(0.0, sigma)
    order = eligible[np.argsort(latent)[::-1]]
    return order[: min(count, len(order))]


def _core_role_probabilities(rng: np.random.Generator, core_candidates: np.ndarray) -> np.ndarray:
    count = len(core_candidates)
    if count <= 0:
        return np.array([], dtype=float)
    if count == 1:
        return np.array([1.0])
    if count == 2:
        center = np.array([0.66, 0.34])
        concentration = 38.0
    else:
        center = np.array([0.57, 0.29, 0.14])
        concentration = 32.0
    return rng.dirichlet(center * concentration)


def sample_event_rush_share_plan(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Sample one stable designed-rushing hierarchy for an event-sim game world.

    v1.3 chooses a rusher on each RUN event, so feeding it static season shares makes
    every fringe player repeatedly eligible and flattens feature-back workloads. This
    bridge reuses the already-governed Role A/B/C hierarchy at the *game* level:

    - QB designed-run mass is reserved from the current transfer-safe QB share prior.
    - Non-QB roles are selected once for the game using current availability, role
      probability, uncertainty, effectiveness and historical/current rush share.
    - Core roles own the finite-workload mass; a small incidental reservoir remains for
      gadget/peripheral carriers.

    The function returns shares only. It does not create carries or alter pass/run play
    calling, so team rushing volume still emerges entirely from the v1.3 football game.
    """
    players = pool.players
    if not players:
        return {}

    base = np.asarray([max(player.rush_share, 0.0) for player in players], dtype=float)
    if base.sum() <= 0:
        return {}
    base /= base.sum()

    positions = np.asarray([player.position.upper() for player in players])
    qb_mask = positions == "QB"
    non_qb_mask = ~qb_mask
    qb_mass = float(np.clip(base[qb_mask].sum(), 0.0, 0.45))
    non_qb_mass = 1.0 - qb_mass

    out = np.zeros(len(players), dtype=float)
    if qb_mask.any() and qb_mass > 0:
        qb_base = base[qb_mask]
        qb_total = float(qb_base.sum())
        if qb_total > 0:
            out[qb_mask] = qb_mass * qb_base / qb_total

    non_qb_indices = np.flatnonzero(non_qb_mask & (base > 0))
    if len(non_qb_indices) == 0 or non_qb_mass <= 0:
        total = float(out.sum())
        return {
            player.player_id: float(out[idx] / total)
            for idx, player in enumerate(players)
            if out[idx] > 0 and total > 0
        }

    active_probability = np.asarray(
        [player.active_probability for player in players], dtype=float
    )
    role_probability = np.asarray(
        [player.rush_role_probability for player in players], dtype=float
    )
    role_uncertainty = np.asarray(
        [player.role_uncertainty for player in players], dtype=float
    )
    effectiveness = np.asarray(
        [player.effectiveness_if_active for player in players], dtype=float
    )

    active = rng.random(len(players)) < np.clip(active_probability, 0.0, 1.0)
    eligible = non_qb_indices[active[non_qb_indices]]
    if len(eligible) == 0:
        eligible = np.array([non_qb_indices[np.argmax(base[non_qb_indices])]], dtype=int)

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

    expected_total_runs = int(
        np.clip(
            round(expected_scrimmage_plays * (1.0 - pool.neutral_pass_rate)),
            8,
            38,
        )
    )
    expected_non_qb_runs = max(round(expected_total_runs * non_qb_mass), 1)
    core_count = _sample_core_count(rng, expected_non_qb_runs, len(eligible))
    core = _sample_core_roles(
        rng,
        eligible,
        role_strength,
        role_uncertainty,
        core_count,
    )
    if len(core) == 0:
        core = np.array([eligible[np.argmax(base[eligible])]], dtype=int)
    core_p = _core_role_probabilities(rng, core)

    peripheral = np.setdiff1d(eligible, core, assume_unique=False)
    incidental_mass = INCIDENTAL_RUSH_SHARE if len(peripheral) else 0.0
    out[core] += non_qb_mass * (1.0 - incidental_mass) * core_p
    if len(peripheral):
        weights = np.clip(peripheral_weight[peripheral], 0.0, None)
        if weights.sum() <= 0:
            weights = np.clip(base[peripheral], 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(len(peripheral), dtype=float)
        weights /= weights.sum()
        out[peripheral] += non_qb_mass * incidental_mass * weights

    total = float(out.sum())
    if total <= 0:
        return {}
    out /= total
    return {
        player.player_id: float(out[idx])
        for idx, player in enumerate(players)
        if out[idx] > 0
    }


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

    qb_indices = np.array([idx for idx, p in enumerate(players) if p.position == "QB"], dtype=int)
    non_qb_indices = np.array([idx for idx, p in enumerate(players) if p.position != "QB"], dtype=int)
    reserved_qb_attempts = np.zeros((worlds, n_players), dtype=np.int16)
    reserved_qb_tds = np.zeros((worlds, n_players), dtype=np.int16)
    for idx in qb_indices:
        stats = side.player_stats[players[idx].player_id]
        reserved_qb_attempts[:, idx] = stats["rush_attempts"].astype(np.int16)
        reserved_qb_tds[:, idx] = stats["rushing_tds"].astype(np.int16)

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
            # Weight vectors passed to choice must describe exactly the candidate set.
            # Using the full-roster vector here made small synthetic/player pools fail and,
            # more importantly, could decouple incidental work from the actual eligible set.
            chosen = _weighted_choice_without_replacement(
                rng, peripheral, peripheral_weight[peripheral], incidental_count
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

    for w in range(worlds):
        qb_carries = int(reserved_qb_attempts[w].sum())
        team_td_supply = int(team_rush_tds[w])
        qb_td_claims = reserved_qb_tds[w, qb_indices].astype(int)
        qb_td_total = int(qb_td_claims.sum())
        if qb_td_total > team_td_supply and len(qb_indices):
            weights = qb_td_claims.astype(float)
            if weights.sum() <= 0:
                weights = reserved_qb_attempts[w, qb_indices].astype(float)
            if weights.sum() <= 0:
                weights = np.ones(len(qb_indices), dtype=float)
            weights /= weights.sum()
            reserved_qb_tds[w, qb_indices] = rng.multinomial(
                team_td_supply, weights
            ).astype(np.int16)
            qb_td_total = team_td_supply

        residual_carries = max(int(team_rushes[w]) - qb_carries, 0)
        residual_tds = max(team_td_supply - qb_td_total, 0)

        non_qb_weights = attempts_out[w, non_qb_indices].astype(float)
        if len(non_qb_indices) and residual_carries > 0:
            if non_qb_weights.sum() <= 0:
                non_qb_weights = np.clip(base[non_qb_indices], 0.0, None)
            if non_qb_weights.sum() <= 0:
                non_qb_weights = np.ones(len(non_qb_indices), dtype=float)
            non_qb_weights /= non_qb_weights.sum()
            attempts_out[w, non_qb_indices] = rng.multinomial(
                residual_carries, non_qb_weights
            ).astype(np.int16)
        elif len(non_qb_indices):
            attempts_out[w, non_qb_indices] = 0

        attempts_out[w, qb_indices] = reserved_qb_attempts[w, qb_indices]

        if len(non_qb_indices) and residual_tds > 0:
            td_weights = attempts_out[w, non_qb_indices].astype(float) * np.array(
                [max(players[idx].red_zone_rush_share, players[idx].rushing_td_share, 0.01)
                 for idx in non_qb_indices],
                dtype=float,
            )
            if td_weights.sum() <= 0:
                td_weights = np.ones(len(non_qb_indices), dtype=float)
            td_weights /= td_weights.sum()
            rush_td_out[w, non_qb_indices] = rng.multinomial(
                residual_tds, td_weights
            ).astype(np.int16)
        elif len(non_qb_indices):
            rush_td_out[w, non_qb_indices] = 0
        rush_td_out[w, qb_indices] = reserved_qb_tds[w, qb_indices]

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
