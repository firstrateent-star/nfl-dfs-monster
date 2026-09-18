from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import reality_v62
from monster.sim.rushing_roles import sample_event_rush_share_plan as _native_rush_plan
from monster.snapshot.player import TeamPlayerPool


def _is_current_rb_starter(player) -> bool:
    return (
        player.position.upper() == "RB"
        and player.active_probability >= 0.50
        and v635._is_current_starter(player.player_id)
    )


def sample_rush_share_plan_v636(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Reallocate only the RB portion of the native rushing plan.

    v6.3.5 correctly preserved QB rushing mass, but it applied current depth/snap
    evidence to every non-QB participant. That made ordinary WR1/TE1 snap
    participation behave like designed-rushing evidence and inflated gadget
    carry mass.

    v6.3.6 separates the causal channels:
    - QB mass stays exactly as sampled by the native v6.3 rushing hierarchy.
    - WR/TE gadget mass and within-group shares stay exactly native.
    - Current depth/snap evidence may reallocate only the RB workload.
    - A current RB starter may enter despite sparse/zero old rushing history.

    This changes who owns RB carries without manufacturing carries for players
    merely because they are receiving-game starters.
    """

    base = _native_rush_plan(
        pool,
        rng=rng,
        expected_scrimmage_plays=expected_scrimmage_plays,
    )
    if not base:
        return {}

    by_id = {player.player_id: player for player in pool.players}
    qb_ids = {
        player.player_id for player in pool.players if player.position.upper() == "QB"
    }
    rb_ids = {
        player.player_id for player in pool.players if player.position.upper() == "RB"
    }
    gadget_ids = {
        player.player_id
        for player in pool.players
        if player.position.upper() in {"WR", "TE"}
    }

    # These masses are already governed by the native rushing-role model. Current
    # receiving participation is not new evidence that they should be redesigned.
    preserved_ids = qb_ids | gadget_ids
    preserved_mass = sum(float(base.get(player_id, 0.0)) for player_id in preserved_ids)
    rb_mass = max(1.0 - preserved_mass, 0.0)

    participant_ids = [
        player_id
        for player_id, share in base.items()
        if player_id in rb_ids and float(share) > 0.0
    ]
    for player in pool.players:
        if (
            player.player_id in rb_ids
            and player.player_id not in participant_ids
            and _is_current_rb_starter(player)
        ):
            participant_ids.append(player.player_id)

    if not participant_ids or rb_mass <= 0.0:
        return base

    weights: list[float] = []
    uncertainties: list[float] = []
    for player_id in participant_ids:
        player = by_id[player_id]
        prior_share = max(float(base.get(player_id, 0.0)), 0.0)
        current = v635._current_role_signal(player_id)
        conditional = max(prior_share, 1e-6) ** 0.82 * (0.68 + 0.64 * current)
        if prior_share < 0.02 and _is_current_rb_starter(player):
            conditional += 0.045 * current
        weights.append(max(conditional, 1e-6))
        uncertainties.append(float(np.clip(player.role_uncertainty, 0.02, 0.35)))

    centers = np.asarray(weights, dtype=float)
    centers /= centers.sum()
    centers = reality_v62._cap_simplex(
        centers,
        v635._role_share_cap(len(participant_ids), base=0.82),
    )
    concentration = float(
        np.clip(58.0 - 70.0 * float(np.mean(uncertainties)), 26.0, 54.0)
    )
    shares = rng.dirichlet(np.clip(centers * concentration, 0.25, None))
    shares = reality_v62._cap_simplex(
        shares,
        v635._role_share_cap(len(participant_ids), base=0.88),
    )

    out = {
        player_id: float(base[player_id])
        for player_id in preserved_ids
        if float(base.get(player_id, 0.0)) > 0.0
    }
    for index, player_id in enumerate(participant_ids):
        out[player_id] = rb_mass * float(shares[index])

    total = sum(out.values())
    if total <= 0.0:
        return base
    return {
        key: value / total
        for key, value in out.items()
        if value > 0.0
    }


def install_current_role_guard_v636(personnel: pl.DataFrame) -> None:
    """Keep v6.3.5 target logic and replace only rushing workload translation."""

    v635.configure_current_skill_roles_v635(personnel)
    reality_v62.receiver_candidate_ids = v635.receiver_candidate_ids_v635
    reality_v62.sample_target_share_plan = v635.sample_target_share_plan_v635
    reality_v62.sample_event_rush_share_plan = sample_rush_share_plan_v636
