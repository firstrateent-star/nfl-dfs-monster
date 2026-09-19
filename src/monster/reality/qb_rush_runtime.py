from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monster.reality.qb_rush_authority import (
    apply_qb_family_role_world,
    separate_qb_from_rush_role_plan,
)
from monster.sim.reality_v62 import RoleWorldPlan

RoleSampler = Callable[..., object]
RoleApplier = Callable[..., object]

_BASE_ROLE_SAMPLER: RoleSampler | None = None
_BASE_ROLE_APPLIER: RoleApplier | None = None


def configure_base_role_hooks(
    *,
    sampler: RoleSampler,
    applier: RoleApplier,
) -> None:
    """Capture the inherited role runtime before v7.1 installs its wrappers."""

    global _BASE_ROLE_APPLIER, _BASE_ROLE_SAMPLER
    _BASE_ROLE_SAMPLER = sampler
    _BASE_ROLE_APPLIER = applier


def base_role_hooks_ready() -> bool:
    return _BASE_ROLE_SAMPLER is not None and _BASE_ROLE_APPLIER is not None


def sample_role_world_v701(pool: Any, *, rng: Any, **kwargs: Any) -> object:
    """Preserve the inherited role draw, then remove mixed QB rush-share mass.

    No extra RNG calls are made here. That keeps common-random-number pairing
    intact and makes the QB-family change mechanically isolated.
    """

    if _BASE_ROLE_SAMPLER is None:
        raise RuntimeError("v7.1 base role sampler has not been installed")
    plan = _BASE_ROLE_SAMPLER(pool, rng=rng, **kwargs)
    if not isinstance(plan, RoleWorldPlan):
        return plan
    return separate_qb_from_rush_role_plan(pool, plan)


def apply_role_world_v701(team: Any, plan: object) -> object:
    """Apply the QB-separated rush world while retaining inherited target roles."""

    if isinstance(plan, RoleWorldPlan):
        return apply_qb_family_role_world(team, plan)
    if _BASE_ROLE_APPLIER is None:
        raise RuntimeError("v7.1 base role applier has not been installed")
    return _BASE_ROLE_APPLIER(team, plan)
