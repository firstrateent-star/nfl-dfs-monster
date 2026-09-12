from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.play_kernel import PlayerIdentity


@dataclass(frozen=True)
class RichPlayerIdentity(PlayerIdentity):
    """Play-level player identity that preserves mechanism-specific capability.

    The legacy ``PlayerIdentity`` fields remain for compatibility with the existing v1.3
    engine.  These additional channels are signed relative signals in [-1, 1] compiled from
    Madden/physical/NFL evidence.  Snap mechanisms consume only channels with football
    jurisdiction instead of collapsing every skill into one generic efficiency multiplier.
    """

    speed_skill: float = 0.0
    mobility_skill: float = 0.0
    qb_execution_skill: float = 0.0
    route_separation_skill: float = 0.0
    catchpoint_skill: float = 0.0
    rush_creation_skill: float = 0.0
    runner_power_skill: float = 0.0
    open_field_skill: float = 0.0
    ball_security_skill: float = 0.0
    evidence_fields: int = 0

    def channel(self, name: str) -> float:
        value = getattr(self, f"{name}_skill", 0.0)
        return float(np.clip(value, -1.0, 1.0))


def skill_multiplier(player: object, channel: str, authority: float) -> float:
    """Convert one signed identity channel into a bounded neutral-centered multiplier."""

    value = float(np.clip(getattr(player, f"{channel}_skill", 0.0), -1.0, 1.0))
    return float(np.clip(1.0 + authority * value, 1.0 - authority, 1.0 + authority))


def route_skill(player: object) -> float:
    """Route/separation capability used only by coverage interaction."""

    return float(
        np.clip(
            float(getattr(player, "efficiency", 1.0))
            * skill_multiplier(player, "route_separation", 0.18)
            * skill_multiplier(player, "speed", 0.06),
            0.55,
            1.55,
        )
    )


def catch_skill(player: object) -> float:
    """Catch-point capability used only by throw/catch resolution."""

    return float(
        np.clip(
            float(getattr(player, "efficiency", 1.0))
            * skill_multiplier(player, "catchpoint", 0.16),
            0.55,
            1.55,
        )
    )


def rush_creation_skill(player: object) -> float:
    """Vision/cut/creation capability used by run-lane resolution."""

    return float(
        np.clip(
            float(getattr(player, "efficiency", 1.0))
            * skill_multiplier(player, "rush_creation", 0.18),
            0.55,
            1.55,
        )
    )


def open_field_skill(player: object) -> float:
    """Open-field/explosive capability used after space is created."""

    return float(
        np.clip(
            float(getattr(player, "explosive", 1.0))
            * skill_multiplier(player, "open_field", 0.20)
            * skill_multiplier(player, "speed", 0.08),
            0.55,
            1.60,
        )
    )


def qb_execution_skill(player: object) -> float:
    """Read/throw/pressure-processing capability for QB decision resolution."""

    return float(
        np.clip(
            float(getattr(player, "efficiency", 1.0))
            * skill_multiplier(player, "qb_execution", 0.18),
            0.55,
            1.55,
        )
    )


def qb_mobility_skill(player: object) -> float:
    """Escape/scramble capability, distinct from passing execution."""

    return float(
        np.clip(
            float(getattr(player, "explosive", 1.0))
            * skill_multiplier(player, "mobility", 0.22)
            * skill_multiplier(player, "speed", 0.08),
            0.55,
            1.60,
        )
    )
