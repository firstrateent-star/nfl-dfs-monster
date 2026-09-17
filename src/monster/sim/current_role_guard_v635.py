from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from monster.sim import reality_v62
from monster.sim.rushing_roles import sample_event_rush_share_plan as _native_rush_plan
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class CurrentSkillRoleV635:
    position: str
    depth_rank: int
    conditional_offense_snap_share: float
    active_probability: float
    status: str


_CURRENT_ROLES: dict[str, CurrentSkillRoleV635] = {}


def configure_current_skill_roles_v635(personnel: pl.DataFrame) -> None:
    """Cache current depth/snap truth separately from historical opportunity priors."""

    global _CURRENT_ROLES
    roles: dict[str, CurrentSkillRoleV635] = {}
    for row in personnel.to_dicts():
        position = str(row.get("position") or "").upper()
        if position not in {"QB", "RB", "WR", "TE"}:
            continue
        player_id = str(row.get("gsis_id") or row.get("pfr_id") or row.get("display_name") or "")
        if not player_id:
            continue
        try:
            depth_rank = int(float(row.get("depth_rank") or 0))
        except (TypeError, ValueError):
            depth_rank = 0
        try:
            snap = float(row.get("conditional_offense_snap_share") or 0.0)
        except (TypeError, ValueError):
            snap = 0.0
        try:
            active = float(row.get("game_day_active_probability") or 0.0)
        except (TypeError, ValueError):
            active = 0.0
        roles[player_id] = CurrentSkillRoleV635(
            position=position,
            depth_rank=depth_rank,
            conditional_offense_snap_share=float(np.clip(snap, 0.0, 1.0)),
            active_probability=float(np.clip(active, 0.0, 1.0)),
            status=str(row.get("status") or "").upper(),
        )
    _CURRENT_ROLES = roles


def _current_role_signal(player_id: str) -> float:
    role = _CURRENT_ROLES.get(str(player_id))
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
    # Depth determines participation authority; conditional snap evidence differentiates
    # current starters/backups without replacing historical opportunity hierarchy.
    return float(
        np.clip(
            role.active_probability
            * (0.55 * depth + 0.45 * role.conditional_offense_snap_share),
            0.0,
            1.0,
        )
    )


def _is_current_starter(player_id: str) -> bool:
    role = _CURRENT_ROLES.get(str(player_id))
    if role is None:
        return False
    return (
        role.status == "ACT"
        and role.depth_rank == 1
        and role.active_probability >= 0.50
        and role.conditional_offense_snap_share >= 0.12
    )


def _role_share_cap(player_count: int, *, base: float) -> float:
    if player_count <= 1:
        return 1.0
    if player_count == 2:
        return max(base, 0.86)
    if player_count == 3:
        return max(base, 0.66)
    return base


def receiver_candidate_ids_v635(
    pool: TeamPlayerPool,
    *,
    max_candidates: int = 10,
) -> tuple[str, ...]:
    """Current truth owns eligibility; history owns most of the rank within that universe."""

    eligible = [
        player
        for player in pool.players
        if player.position.upper() in {"RB", "WR", "TE"}
        and player.active_probability >= 0.10
    ]
    if not eligible:
        return ()

    protected = [player for player in eligible if _is_current_starter(player.player_id)]
    protected_ids = {player.player_id for player in protected}
    others = [player for player in eligible if player.player_id not in protected_ids]
    ranked = sorted(
        others,
        key=lambda player: (
            max(float(player.target_share), 0.0) ** 0.78
            + 0.018
            * _current_role_signal(player.player_id)
            * (0.35 + float(player.role_uncertainty))
            * reality_v62._receiving_skill(player)
        )
        * float(np.clip(player.active_probability, 0.0, 1.0)),
        reverse=True,
    )
    room = max(max_candidates - len(protected), 0)
    selected = protected + ranked[:room]
    order = {player.player_id: index for index, player in enumerate(pool.players)}
    selected.sort(key=lambda player: order.get(player.player_id, 10_000))
    return tuple(player.player_id for player in selected)


def _target_center(player, selected_count: int) -> float:
    history = max(float(player.target_share), 0.0)
    # Current role supplies reserve only when history is sparse. Existing alpha players do not
    # all receive the same additive reserve, which was the v6.3.4 alpha-dilution failure mode.
    sparse = max(0.02 - history, 0.0) / 0.02
    reserve = (
        reality_v62.TARGET_ROLE_RESERVE
        / max(selected_count, 1)
        * sparse
        * _current_role_signal(player.player_id)
        * (0.40 + float(player.role_uncertainty))
        * reality_v62._receiving_skill(player)
    )
    current_multiplier = 0.94 + 0.12 * _current_role_signal(player.player_id)
    return max(history, 0.0) ** 0.78 * current_multiplier + reserve


def sample_target_share_plan_v635(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    max_active_receivers: int = 8,
) -> dict[str, float]:
    candidate_ids = set(receiver_candidate_ids_v635(pool))
    candidates = [
        player
        for player in pool.players
        if player.player_id in candidate_ids and player.position.upper() in {"RB", "WR", "TE"}
    ]
    if not candidates:
        return {}

    active = [
        player
        for player in candidates
        if rng.random() < float(np.clip(player.active_probability, 0.0, 1.0))
    ]
    if not active:
        active = [max(candidates, key=lambda player: player.active_probability)]

    strength = np.asarray(
        [_target_center(player, len(active)) for player in active],
        dtype=float,
    )
    sigma = np.asarray(
        [0.08 + 0.75 * float(np.clip(player.role_uncertainty, 0.02, 0.35)) for player in active],
        dtype=float,
    )
    latent = np.log(np.clip(strength, 1e-8, None)) + rng.normal(0.0, sigma)

    protected = [player for player in active if _is_current_starter(player.player_id)]
    protected_ids = {player.player_id for player in protected}
    remaining = [
        active[int(index)]
        for index in np.argsort(latent)[::-1]
        if active[int(index)].player_id not in protected_ids
    ]
    keep_n = max(len(protected), min(max_active_receivers, len(active)))
    selected = protected + remaining[: max(keep_n - len(protected), 0)]

    centers = np.asarray(
        [_target_center(player, len(selected)) for player in selected],
        dtype=float,
    )
    if centers.sum() <= 0:
        centers = np.ones(len(selected), dtype=float)
    centers /= centers.sum()
    centers = reality_v62._cap_simplex(
        centers,
        _role_share_cap(len(selected), base=0.44),
    )

    mean_uncertainty = float(np.mean([player.role_uncertainty for player in selected]))
    concentration = float(np.clip(62.0 - 85.0 * mean_uncertainty, 22.0, 56.0))
    shares = rng.dirichlet(np.clip(centers * concentration, 0.20, None))
    shares = reality_v62._cap_simplex(
        shares,
        _role_share_cap(len(selected), base=0.52),
    )
    return {
        player.player_id: float(shares[index])
        for index, player in enumerate(selected)
        if shares[index] > 0.0
    }


def sample_rush_share_plan_v635(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    expected_scrimmage_plays: float = 62.0,
) -> dict[str, float]:
    """Separate rushing participation truth from conditional workload hierarchy.

    v6.3's finite rushing core remains the prior. Current depth/snap evidence reweights the
    conditional share among participating non-QBs and can admit an active current starter that
    sparse old history omitted. QB rushing mass is preserved exactly.
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
    qb_mass = sum(float(base.get(player_id, 0.0)) for player_id in qb_ids)
    non_qb_mass = max(1.0 - qb_mass, 0.0)

    participant_ids = [
        player_id
        for player_id, share in base.items()
        if player_id not in qb_ids and float(share) > 0.0
    ]
    for player in pool.players:
        if (
            player.position.upper() in {"RB", "WR", "TE"}
            and player.player_id not in participant_ids
            and _is_current_starter(player.player_id)
            and player.active_probability >= 0.50
        ):
            participant_ids.append(player.player_id)

    if not participant_ids or non_qb_mass <= 0:
        return base

    weights: list[float] = []
    uncertainties: list[float] = []
    for player_id in participant_ids:
        player = by_id[player_id]
        prior_share = max(float(base.get(player_id, 0.0)), 0.0)
        current = _current_role_signal(player_id)
        conditional = max(prior_share, 1e-6) ** 0.82 * (0.68 + 0.64 * current)
        # Only truly sparse current starters receive an admission reserve.
        if prior_share < 0.02 and _is_current_starter(player_id):
            conditional += 0.045 * current
        weights.append(max(conditional, 1e-6))
        uncertainties.append(float(np.clip(player.role_uncertainty, 0.02, 0.35)))

    centers = np.asarray(weights, dtype=float)
    centers /= centers.sum()
    centers = reality_v62._cap_simplex(
        centers,
        _role_share_cap(len(participant_ids), base=0.82),
    )
    concentration = float(
        np.clip(58.0 - 70.0 * float(np.mean(uncertainties)), 26.0, 54.0)
    )
    shares = rng.dirichlet(np.clip(centers * concentration, 0.25, None))
    shares = reality_v62._cap_simplex(
        shares,
        _role_share_cap(len(participant_ids), base=0.88),
    )

    out = {
        player_id: float(base[player_id])
        for player_id in qb_ids
        if float(base.get(player_id, 0.0)) > 0.0
    }
    for index, player_id in enumerate(participant_ids):
        out[player_id] = non_qb_mass * float(shares[index])

    total = sum(out.values())
    if total <= 0:
        return base
    return {key: value / total for key, value in out.items() if value > 0.0}


def install_current_role_guard_v635(personnel: pl.DataFrame) -> None:
    configure_current_skill_roles_v635(personnel)
    reality_v62.receiver_candidate_ids = receiver_candidate_ids_v635
    reality_v62.sample_target_share_plan = sample_target_share_plan_v635
    reality_v62.sample_event_rush_share_plan = sample_rush_share_plan_v635
