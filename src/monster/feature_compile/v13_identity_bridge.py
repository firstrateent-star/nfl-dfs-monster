from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.sim.play_kernel import PlayerIdentity


@dataclass(frozen=True)
class V13IdentityTrace:
    speed: float
    catch_skill: float
    power: float
    health: float
    continuity: float
    evidence_fields: int


def _signal(value: float | None, center: float, scale: float, reverse: bool = False) -> float:
    if value is None:
        return 0.0
    raw = (float(value) - center) / max(scale, 1e-6)
    if reverse:
        raw *= -1.0
    return float(np.tanh(raw))


def compile_v13_player_identity(
    *,
    player_id: str,
    name: str,
    position: str,
    usage_weight: float,
    inputs: PlayerMechanismInputs,
    availability_already_sampled: bool = False,
) -> tuple[PlayerIdentity, V13IdentityTrace]:
    """Compile Full-Reality evidence into bounded play-level identity traits.

    Missing evidence is neutral. These traits have football-event jurisdiction only and never
    receive market, salary, ownership, optimizer or fantasy authority.

    ``availability_already_sampled`` is the seam for world-specific personnel reality. When
    false, the legacy bridge preserves its historical expected-state behavior. When true, the
    caller has already decided whether the player exists in this game world, so active
    probability MUST NOT reduce that player's ability again. Only effectiveness-if-active may
    alter active-world capability.
    """
    speed_parts = [
        _signal(inputs.forty_time, 4.55, 0.18, reverse=True),
        _signal(inputs.madden_speed, 85.0, 8.0),
        _signal(inputs.madden_acceleration, 85.0, 8.0),
    ]
    speed = float(np.mean(speed_parts))
    catch = float(
        np.mean(
            [
                _signal(inputs.madden_catching, 82.0, 10.0),
                _signal(inputs.madden_route_running, 82.0, 10.0),
                _signal(inputs.height_in, 73.0, 4.0),
                _signal(inputs.wingspan_in, 78.0, 5.0),
            ]
        )
    )
    power = float(
        np.mean(
            [
                _signal(inputs.weight_lbs, 215.0, 28.0),
                _signal(inputs.height_in, 73.0, 4.0),
            ]
        )
    )
    health = 1.0
    if inputs.effectiveness_if_active is not None:
        health *= float(np.clip(inputs.effectiveness_if_active, 0.35, 1.10))
    if not availability_already_sampled and inputs.active_probability is not None:
        health *= float(np.clip(inputs.active_probability, 0.0, 1.0))
    continuity = (
        0.0
        if inputs.unit_continuity is None
        else float(np.clip((inputs.unit_continuity - 0.5) * 2.0, -1.0, 1.0))
    )

    efficiency = float(
        np.clip(health * (1.0 + 0.035 * catch + 0.025 * continuity), 0.35, 1.10)
    )
    explosive = float(np.clip(1.0 + 0.07 * speed, 0.90, 1.10))
    turnover_security = float(np.clip(1.0 + 0.025 * power, 0.94, 1.06))
    evidence_fields = sum(
        value is not None
        for value in (
            inputs.height_in,
            inputs.weight_lbs,
            inputs.wingspan_in,
            inputs.forty_time,
            inputs.madden_speed,
            inputs.madden_acceleration,
            inputs.madden_route_running,
            inputs.madden_catching,
            inputs.age_years,
            inputs.career_workload,
            inputs.unit_continuity,
            inputs.active_probability,
            inputs.effectiveness_if_active,
        )
    )
    identity = PlayerIdentity(
        player_id=player_id,
        name=name,
        position=position,
        usage_weight=max(float(usage_weight), 0.001),
        efficiency=efficiency,
        explosive=explosive,
        turnover_security=turnover_security,
    )
    return identity, V13IdentityTrace(speed, catch, power, health, continuity, evidence_fields)
