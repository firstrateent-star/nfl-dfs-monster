from __future__ import annotations

import copy

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.sim.gadget_rush_entry_priors_v639 import empirical_gadget_entry_prior
from monster.sim.gadget_rush_priors_v638 import sample_conditional_gadget_carries
from monster.snapshot.player import TeamPlayerPool


def _rotation_exposure_v639(player_id: str) -> float:
    """Current offensive-rotation exposure, separate from gadget identity."""

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

    rotation = 0.55 * depth + 0.45 * role.conditional_offense_snap_share
    return float(np.sqrt(np.clip(rotation, 0.0, 1.0)))


def _entry_probability_v639(player) -> float:
    """Calibrated recurrence prior times current offensive-rotation exposure."""

    prior = empirical_gadget_entry_prior(
        player.position,
        player.historical_rushes,
    )
    return float(
        np.clip(
            prior * _rotation_exposure_v639(player.player_id),
            0.0,
            1.0,
        )
    )


def sample_rush_share_plan_v639(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Use historical gadget recurrence for entry and empirical carries for workload.

    v6.3.9 keeps the v6.3.8 state decomposition but replaces its remaining heuristic
    entry score with direct multi-season recurrence evidence:

    current rotation exposure
        -> empirical entry probability from prior-season carries
        -> empirical conditional carry count
        -> residual workload returned to frozen v6.3.6 RB hierarchy.

    QB rushing mass, RB within-group ordering, target worlds, coaching, matchup
    identity and player execution remain frozen. Production RNG consumption is
    exactly the frozen v6.3.6 sequence; all challenger-only uncertainty is private.
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
        if shadow_rng.random() >= _entry_probability_v639(player):
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


def install_current_role_guard_v639(personnel: pl.DataFrame) -> None:
    """Keep v6.3.6 identity/targets and install calibrated gadget-role recurrence."""

    v636.install_current_role_guard_v636(personnel)
    reality_v62.receiver_candidate_ids = v635.receiver_candidate_ids_v635
    reality_v62.sample_target_share_plan = v635.sample_target_share_plan_v635
    reality_v62.sample_event_rush_share_plan = sample_rush_share_plan_v639
