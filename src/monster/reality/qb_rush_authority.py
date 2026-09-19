from __future__ import annotations

from dataclasses import replace

from monster.snapshot.player import TeamPlayerPool
from monster.sim.play_kernel import TeamIdentity
from monster.sim.reality_v62 import RoleWorldPlan


QB_SENTINEL_USAGE = 0.001


def separate_qb_from_rush_role_plan(
    pool: TeamPlayerPool,
    plan: RoleWorldPlan,
) -> RoleWorldPlan:
    """Remove QB total-rush history from the non-QB designed workload simplex.

    Historical rush share can contain designed runs, scrambles and kneels. v7
    therefore refuses to let the aggregate QB rushing share reserve designed
    workload. Non-QB relative shares are preserved and renormalized. The QB can
    still receive a designed run later through explicit run-geometry evidence.
    """

    positions = {
        str(player.player_id): str(player.position).upper()
        for player in pool.players
    }
    non_qb = {
        str(player_id): float(share)
        for player_id, share in plan.items()
        if positions.get(str(player_id)) != "QB" and float(share) > 0.0
    }
    total = sum(non_qb.values())
    normalized = (
        {}
        if total <= 0.0
        else {
            player_id: share / total
            for player_id, share in non_qb.items()
        }
    )
    return RoleWorldPlan(normalized, plan.target_plan)


def apply_qb_family_role_world(
    team: TeamIdentity,
    plan: RoleWorldPlan,
) -> TeamIdentity:
    """Apply non-QB role ownership while retaining QB concept eligibility.

    The quarterback is appended with a sentinel usage value only so the live
    snap/intent layer can see him. v7 target/rusher assignment must never use
    this sentinel as opportunity authority.
    """

    rushers = tuple(
        replace(rusher, usage_weight=float(plan[rusher.player_id]))
        for rusher in team.rushers
        if rusher.position.upper() != "QB" and plan.get(rusher.player_id, 0.0) > 0.0
    )
    qb = replace(team.quarterback, usage_weight=QB_SENTINEL_USAGE)
    if not rushers:
        rushers = tuple(
            replace(
                rusher,
                usage_weight=max(float(rusher.usage_weight), QB_SENTINEL_USAGE),
            )
            for rusher in team.rushers
            if rusher.position.upper() != "QB"
        )
    return replace(team, rushers=(*rushers, qb))


def designed_qb_geometry_attempts(team: TeamIdentity) -> int:
    """Return direct historical designed-run evidence attached to this QB."""

    ecology = team.intent_ecology
    if ecology is None:
        return 0
    quarterback_id = str(team.quarterback.player_id)
    return sum(
        max(int(attempts), 0)
        for (actor_id, _category), attempts in ecology.rusher_geometry_attempts.items()
        if str(actor_id) == quarterback_id
    )
