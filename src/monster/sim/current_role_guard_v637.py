from __future__ import annotations

import copy

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.snapshot.player import TeamPlayerPool


def sample_rush_share_plan_v637(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Make WR/TE designed-rush entry obey the existing role-probability signal.

    v6.3.6 fixed a category error where receiving starter status could manufacture
    carries, but it deliberately preserved the native WR/TE gadget reservoir. The
    remaining native path has a different causal leak: rush_role_probability is
    defined as the probability an active player enters the designed-carry tree, yet
    the event-plan sampler only uses it as a weight. Because every WR/TE owns a small
    nonzero positional rush prior, nearly every active receiver retains some gadget
    mass in every world.

    v6.3.7 changes only that admission boundary:
    - build the exact v6.3.6 rushing plan first;
    - for WR/TE players already active in that plan, sample designed-rush entry from
      their existing rush_role_probability;
    - rejected gadget mass returns to the existing RB hierarchy proportionally;
    - QB mass and RB within-group ordering/shares are otherwise unchanged;
    - all new randomness lives on a private stream, preserving the downstream target
      random world exactly relative to the frozen v6.3.6 control.
    """

    shadow_rng = np.random.default_rng()
    shadow_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    v636.sample_rush_share_plan_v636(
        pool,
        rng=rng,
        expected_scrimmage_plays=expected_scrimmage_plays,
    )
    base = v636.sample_rush_share_plan_v636(
        pool,
        rng=shadow_rng,
        expected_scrimmage_plays=expected_scrimmage_plays,
    )
    if not base:
        return {}

    by_id = {player.player_id: player for player in pool.players}
    gadget_ids = [
        player_id
        for player_id, share in base.items()
        if float(share) > 0.0
        and player_id in by_id
        and by_id[player_id].position.upper() in {"WR", "TE"}
    ]
    if not gadget_ids:
        return base

    rejected: list[str] = []
    for player_id in gadget_ids:
        probability = float(
            np.clip(by_id[player_id].rush_role_probability, 0.0, 1.0)
        )
        if shadow_rng.random() >= probability:
            rejected.append(player_id)

    if not rejected:
        return base

    rejected_mass = sum(float(base[player_id]) for player_id in rejected)
    if rejected_mass <= 0.0:
        return base

    rb_ids = [
        player_id
        for player_id, share in base.items()
        if float(share) > 0.0
        and player_id in by_id
        and by_id[player_id].position.upper() == "RB"
    ]
    rb_mass = sum(float(base[player_id]) for player_id in rb_ids)

    if not rb_ids or rb_mass <= 0.0:
        return base

    out = {
        player_id: float(share)
        for player_id, share in base.items()
        if player_id not in rejected and float(share) > 0.0
    }
    for player_id in rb_ids:
        out[player_id] = float(out.get(player_id, 0.0)) + (
            rejected_mass * float(base[player_id]) / rb_mass
        )

    total = sum(out.values())
    if total <= 0.0:
        return base
    return {
        player_id: share / total
        for player_id, share in out.items()
        if share > 0.0
    }


def install_current_role_guard_v637(personnel: pl.DataFrame) -> None:
    """Keep the v6.3.6 machine and replace only WR/TE designed-rush admission."""

    v636.install_current_role_guard_v636(personnel)
    reality_v62.receiver_candidate_ids = v635.receiver_candidate_ids_v635
    reality_v62.sample_target_share_plan = v635.sample_target_share_plan_v635
    reality_v62.sample_event_rush_share_plan = sample_rush_share_plan_v637
