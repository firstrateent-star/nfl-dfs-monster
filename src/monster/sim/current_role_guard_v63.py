from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from monster.sim import reality_v62
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class CurrentReceivingRole:
    depth_rank: int
    conditional_offense_snap_share: float
    active_probability: float
    status: str


_CURRENT_ROLES: dict[str, CurrentReceivingRole] = {}


def configure_current_receiving_roles(personnel: pl.DataFrame) -> None:
    """Preserve current depth/snap evidence through the role-world boundary.

    Historical usage is allowed to shape *how much* opportunity a player receives, but an
    available current starter cannot become structurally impossible merely because last year's
    target history or a candidate-count cutoff ranks him below an old-role player.
    """
    global _CURRENT_ROLES
    roles: dict[str, CurrentReceivingRole] = {}
    for row in personnel.to_dicts():
        position = str(row.get("position") or "").upper()
        if position not in {"RB", "WR", "TE"}:
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
        roles[player_id] = CurrentReceivingRole(
            depth_rank=depth_rank,
            conditional_offense_snap_share=float(np.clip(snap, 0.0, 1.0)),
            active_probability=float(np.clip(active, 0.0, 1.0)),
            status=str(row.get("status") or "").upper(),
        )
    _CURRENT_ROLES = roles


def _is_current_starter(player_id: str) -> bool:
    role = _CURRENT_ROLES.get(str(player_id))
    if role is None:
        return False
    return (
        role.status == "ACT"
        and role.depth_rank == 1
        and role.active_probability >= 0.50
        and role.conditional_offense_snap_share >= 0.15
    )


def receiver_candidate_ids_current(
    pool: TeamPlayerPool,
    *,
    max_candidates: int = 10,
) -> tuple[str, ...]:
    eligible = [
        player
        for player in pool.players
        if player.position.upper() in {"RB", "WR", "TE"} and player.active_probability >= 0.10
    ]
    if not eligible:
        return ()

    protected = [player for player in eligible if _is_current_starter(player.player_id)]
    protected_ids = {player.player_id for player in protected}
    others = [player for player in eligible if player.player_id not in protected_ids]
    ranked = sorted(
        others,
        key=lambda player: (
            max(float(player.target_share), 0.0) ** 0.70
            + 0.035
            * (0.35 + float(player.role_uncertainty))
            * reality_v62._receiving_skill(player)
        )
        * float(np.clip(player.active_probability, 0.0, 1.0)),
        reverse=True,
    )
    room = max(max_candidates - len(protected), 0)
    selected = protected + ranked[:room]
    # Stable ordering makes diagnostics and seeded worlds reproducible.
    order = {player.player_id: index for index, player in enumerate(pool.players)}
    selected.sort(key=lambda player: order.get(player.player_id, 10_000))
    return tuple(player.player_id for player in selected)


def _role_share_cap(player_count: int, *, base: float) -> float:
    """Cap concentration without mathematically flattening small current rotations.

    A fixed cap below 0.50 forces any two-player simplex to 50/50, which erases genuine
    historical alpha roles. Small rotations therefore receive enough headroom to preserve
    hierarchy; larger rotations keep the tighter v6.3 concentration guard.
    """
    if player_count <= 1:
        return 1.0
    if player_count == 2:
        return max(base, 0.82)
    if player_count == 3:
        return max(base, 0.62)
    return base


def sample_target_share_plan_current(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    max_active_receivers: int = 8,
) -> dict[str, float]:
    candidate_ids = set(receiver_candidate_ids_current(pool))
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
        [
            max(float(player.target_share), 0.0) ** 0.70
            + reality_v62.TARGET_ROLE_RESERVE
            / max(len(active), 1)
            * (0.35 + float(player.role_uncertainty))
            * reality_v62._receiving_skill(player)
            for player in active
        ],
        dtype=float,
    )
    sigma = np.asarray(
        [0.10 + 0.90 * float(np.clip(player.role_uncertainty, 0.02, 0.35)) for player in active],
        dtype=float,
    )
    latent = np.log(np.clip(strength, 1e-7, None)) + rng.normal(0.0, sigma)

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
        [
            max(float(player.target_share), 0.0) ** 0.72
            + reality_v62.TARGET_ROLE_RESERVE
            / max(len(selected), 1)
            * (0.40 + float(player.role_uncertainty))
            * reality_v62._receiving_skill(player)
            for player in selected
        ],
        dtype=float,
    )
    centers /= centers.sum()
    centers = reality_v62._cap_simplex(centers, _role_share_cap(len(selected), base=0.42))
    mean_uncertainty = float(np.mean([player.role_uncertainty for player in selected]))
    concentration = float(np.clip(52.0 - 70.0 * mean_uncertainty, 18.0, 48.0))
    shares = rng.dirichlet(np.clip(centers * concentration, 0.20, None))
    shares = reality_v62._cap_simplex(shares, _role_share_cap(len(selected), base=0.48))
    return {
        player.player_id: float(shares[index])
        for index, player in enumerate(selected)
        if shares[index] > 0.0
    }


def install_current_role_guard(personnel: pl.DataFrame) -> None:
    configure_current_receiving_roles(personnel)
    reality_v62.receiver_candidate_ids = receiver_candidate_ids_current
    reality_v62.sample_target_share_plan = sample_target_share_plan_current
