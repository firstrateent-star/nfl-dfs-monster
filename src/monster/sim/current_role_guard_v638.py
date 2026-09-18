from __future__ import annotations

import copy

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.sim.gadget_rush_priors_v638 import sample_conditional_gadget_carries
from monster.snapshot.player import TeamPlayerPool


def _rotation_exposure_v638(player_id: str) -> float:
    """Current offensive-rotation exposure, separate from rush-role evidence."""

    role = v635._CURRENT_ROLES.get(str(player_id))
    if role is None or role.status != "ACT" or role.active_probability < 0.10:
        return 0.0

    if role.depth_rank == 1:
        depth = 1.0
    elif role.depth_rank == 2:
        depth = 0.62
    elif role.depth_rank == 3:
        depth = 0.36
    else:
        depth = 0.20

    # The active-state draw is already handled by the inherited v6.3.6/native plan.
    # This conditional exposure only answers whether a dressed player is meaningfully
    # in the offensive rotation. Square-root scaling keeps low-snap gadget specialists
    # viable without treating a fringe roster spot like a full-time offensive role.
    rotation = 0.55 * depth + 0.45 * role.conditional_offense_snap_share
    return float(np.sqrt(np.clip(rotation, 0.0, 1.0)))


def _entry_probability_v638(player) -> float:
    return float(
        np.clip(
            float(player.rush_role_probability)
            * _rotation_exposure_v638(player.player_id),
            0.0,
            1.0,
        )
    )


def sample_rush_share_plan_v638(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Decompose gadget rushing into exposure, entry, and conditional workload.

    v6.3.6 is the frozen foundation for QB mass and RB hierarchy. v6.3.8 replaces
    only the WR/TE gadget path:

    1. Current depth/snap truth supplies offensive-rotation exposure.
    2. rush_role_probability supplies designed-rush entry probability only.
    3. Once admitted, conditional carries are sampled from market-blind 2022-25
       position distributions rather than reusing role probability as a share weight.
    4. Gadget carry mass is translated into a share of expected team rushing volume.
    5. The residual non-QB workload returns to the exact v6.3.6 RB hierarchy.

    The production RNG is advanced exactly as v6.3.6 would advance it. All new
    challenger randomness lives on a private stream, so the downstream target world
    remains common-random-number paired to the frozen control.
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
    qb_ids = [
        player.player_id
        for player in pool.players
        if player.position.upper() == "QB"
        and float(base.get(player.player_id, 0.0)) > 0.0
    ]
    rb_ids = [
        player.player_id
        for player in pool.players
        if player.position.upper() == "RB"
        and float(base.get(player.player_id, 0.0)) > 0.0
    ]
    gadget_ids = [
        player.player_id
        for player in pool.players
        if player.position.upper() in {"WR", "TE"}
        and float(base.get(player.player_id, 0.0)) > 0.0
    ]

    qb_mass = sum(float(base.get(player_id, 0.0)) for player_id in qb_ids)
    non_qb_mass = max(1.0 - qb_mass, 0.0)
    rb_base_mass = sum(float(base.get(player_id, 0.0)) for player_id in rb_ids)

    if not rb_ids or rb_base_mass <= 0.0 or non_qb_mass <= 0.0:
        return base

    expected_total_runs = int(
        np.clip(
            round(expected_scrimmage_plays * (1.0 - pool.neutral_pass_rate)),
            8,
            38,
        )
    )

    gadget_shares: dict[str, float] = {}
    for player_id in gadget_ids:
        player = by_id[player_id]
        entry_probability = _entry_probability_v638(player)
        if shadow_rng.random() >= entry_probability:
            continue
        carries = sample_conditional_gadget_carries(
            player.position,
            rng=shadow_rng,
        )
        gadget_shares[player_id] = float(carries) / float(expected_total_runs)

    gadget_mass = sum(gadget_shares.values())
    if gadget_mass >= non_qb_mass and gadget_mass > 0.0:
        scale = (non_qb_mass * 0.98) / gadget_mass
        gadget_shares = {
            player_id: share * scale
            for player_id, share in gadget_shares.items()
        }
        gadget_mass = sum(gadget_shares.values())

    rb_mass = max(non_qb_mass - gadget_mass, 0.0)

    out = {
        player_id: float(base[player_id])
        for player_id in qb_ids
        if float(base.get(player_id, 0.0)) > 0.0
    }
    out.update(
        {
            player_id: float(share)
            for player_id, share in gadget_shares.items()
            if share > 0.0
        }
    )
    for player_id in rb_ids:
        out[player_id] = rb_mass * float(base[player_id]) / rb_base_mass

    total = sum(out.values())
    if total <= 0.0:
        return base
    return {
        player_id: share / total
        for player_id, share in out.items()
        if share > 0.0
    }


def install_current_role_guard_v638(personnel: pl.DataFrame) -> None:
    """Keep v6.3.6 identity/targets and replace only gadget-rush decomposition."""

    v636.install_current_role_guard_v636(personnel)
    reality_v62.receiver_candidate_ids = v635.receiver_candidate_ids_v635
    reality_v62.sample_target_share_plan = v635.sample_target_share_plan_v635
    reality_v62.sample_event_rush_share_plan = sample_rush_share_plan_v638
