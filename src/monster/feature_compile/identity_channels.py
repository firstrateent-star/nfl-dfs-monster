from __future__ import annotations

from dataclasses import dataclass
from math import tanh

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs


@dataclass(frozen=True)
class PlayerIdentityChannels:
    """Source-agnostic football capabilities used by the v1.3 event engine.

    Evidence sources (combine/body measurements, Madden/scouting proxies, future tracking
    metrics, etc.) should map into these mechanism channels rather than directly changing
    points or fantasy output. New evidence can therefore be added without changing the
    scoreboard contract.
    """

    speed: float = 0.0
    mobility: float = 0.0
    qb_execution: float = 0.0
    route_separation: float = 0.0
    catchpoint: float = 0.0
    rush_creation: float = 0.0
    runner_power: float = 0.0
    open_field: float = 0.0
    ball_security: float = 0.0
    evidence_fields: int = 0


_POSITION_REFERENCE = {
    "QB": {"height": 75.0, "weight": 225.0, "forty": 4.80, "wingspan": 79.0},
    "RB": {"height": 70.0, "weight": 210.0, "forty": 4.50, "wingspan": 76.0},
    "WR": {"height": 73.0, "weight": 200.0, "forty": 4.48, "wingspan": 77.0},
    "TE": {"height": 77.0, "weight": 245.0, "forty": 4.70, "wingspan": 80.0},
}


def _signal(
    value: float | None,
    center: float,
    scale: float,
    *,
    reverse: bool = False,
) -> float | None:
    if value is None:
        return None
    raw = (float(value) - center) / max(scale, 1e-6)
    if reverse:
        raw *= -1.0
    return float(tanh(raw))


def _weighted(parts: tuple[tuple[float | None, float], ...]) -> float:
    present = [(value, weight) for value, weight in parts if value is not None and weight > 0.0]
    if not present:
        return 0.0
    total = sum(weight for _, weight in present)
    return float(np.clip(sum(float(value) * weight for value, weight in present) / total, -1.0, 1.0))


def _rating(value: float | None, center: float = 82.0, scale: float = 10.0) -> float | None:
    return _signal(value, center, scale)


def compile_player_identity_channels(
    *,
    position: str,
    physical: PlayerMechanismInputs | None,
    capability: UnitPlayerInputs | None,
) -> PlayerIdentityChannels:
    """Route current human/scouting evidence into football mechanism channels.

    Missing inputs are neutral and are not imputed. The returned values are dimensionless
    relative signals in [-1, 1]; downstream code owns the bounded amount of authority each
    channel receives. This is the extension seam for future 1v1, 11v11 and tracking evidence.
    """

    p = position.upper()
    ref = _POSITION_REFERENCE.get(p, _POSITION_REFERENCE["WR"])
    physical = physical or PlayerMechanismInputs()

    speed_rating = None if capability is None else capability.madden_speed
    accel_rating = None if capability is None else capability.madden_acceleration
    route_rating = None if capability is None else capability.madden_route_running
    catch_rating = None if capability is None else capability.madden_catching

    speed = _weighted(
        (
            (_signal(physical.forty_time, ref["forty"], 0.16, reverse=True), 0.45),
            (_rating(speed_rating, 85.0, 8.0), 0.35),
            (_rating(accel_rating, 85.0, 8.0), 0.20),
        )
    )

    awareness = None if capability is None else capability.madden_awareness
    throw_power = None if capability is None else capability.madden_throw_power
    throw_accuracy = None if capability is None else capability.madden_throw_accuracy
    throw_pressure = None if capability is None else capability.madden_throw_under_pressure
    throw_on_run = None if capability is None else getattr(capability, "madden_throw_on_run", None)
    play_action = None if capability is None else getattr(capability, "madden_play_action", None)
    break_sack = None if capability is None else getattr(capability, "madden_break_sack", None)

    qb_execution = _weighted(
        (
            (_rating(throw_accuracy), 0.34),
            (_rating(throw_pressure), 0.22),
            (_rating(awareness), 0.16),
            (_rating(throw_power), 0.10),
            (_rating(throw_on_run), 0.10),
            (_rating(play_action), 0.08),
        )
    )
    mobility = _weighted(
        (
            (speed, 0.35),
            (_rating(accel_rating, 85.0, 8.0), 0.20),
            (_rating(throw_on_run), 0.15),
            (_rating(break_sack), 0.15),
            (_rating(None if capability is None else getattr(capability, "madden_agility", None)), 0.075),
            (_rating(None if capability is None else getattr(capability, "madden_change_of_direction", None)), 0.075),
        )
    )

    release = None if capability is None else capability.madden_release
    route_separation = _weighted(
        (
            (_rating(route_rating), 0.34),
            (_rating(release), 0.24),
            (speed, 0.16),
            (_rating(accel_rating, 85.0, 8.0), 0.10),
            (_rating(None if capability is None else getattr(capability, "madden_agility", None)), 0.08),
            (_rating(None if capability is None else getattr(capability, "madden_change_of_direction", None)), 0.08),
        )
    )

    catchpoint = _weighted(
        (
            (_rating(catch_rating), 0.30),
            (_rating(None if capability is None else getattr(capability, "madden_catch_in_traffic", None)), 0.18),
            (_rating(None if capability is None else getattr(capability, "madden_spectacular_catch", None)), 0.12),
            (_signal(physical.height_in, ref["height"], 3.0), 0.10),
            (_signal(physical.wingspan_in, ref["wingspan"], 4.0), 0.10),
            (_rating(None if capability is None else getattr(capability, "madden_jump", None)), 0.10),
            (_rating(awareness), 0.10),
        )
    )

    vision = None if capability is None else capability.madden_ball_carrier_vision
    carrying = None if capability is None else capability.madden_carrying
    break_tackle = None if capability is None else capability.madden_break_tackle
    strength = None if capability is None else getattr(capability, "madden_strength", None)
    agility = None if capability is None else getattr(capability, "madden_agility", None)
    change_dir = None if capability is None else getattr(capability, "madden_change_of_direction", None)
    trucking = None if capability is None else getattr(capability, "madden_trucking", None)
    stiff_arm = None if capability is None else getattr(capability, "madden_stiff_arm", None)
    juke = None if capability is None else getattr(capability, "madden_juke", None)
    spin = None if capability is None else getattr(capability, "madden_spin", None)
    stamina = None if capability is None else getattr(capability, "madden_stamina", None)

    rush_creation = _weighted(
        (
            (_rating(vision), 0.24),
            (_rating(break_tackle), 0.18),
            (_rating(accel_rating, 85.0, 8.0), 0.14),
            (_rating(change_dir), 0.12),
            (_rating(agility), 0.10),
            (speed, 0.10),
            (_rating(carrying), 0.07),
            (_rating(awareness), 0.05),
        )
    )
    runner_power = _weighted(
        (
            (_rating(break_tackle), 0.30),
            (_rating(trucking), 0.24),
            (_rating(stiff_arm), 0.14),
            (_rating(strength), 0.12),
            (_signal(physical.weight_lbs, ref["weight"], 24.0), 0.12),
            (_rating(carrying), 0.08),
        )
    )
    open_field = _weighted(
        (
            (speed, 0.24),
            (_rating(accel_rating, 85.0, 8.0), 0.18),
            (_rating(agility), 0.14),
            (_rating(change_dir), 0.14),
            (_rating(juke), 0.10),
            (_rating(spin), 0.06),
            (_rating(break_tackle), 0.09),
            (_rating(vision), 0.05),
        )
    )
    ball_security = _weighted(
        (
            (_rating(carrying), 0.55),
            (_rating(awareness), 0.20),
            (_rating(strength), 0.10),
            (_rating(stamina), 0.10),
            (_rating(break_sack), 0.05),
        )
    )

    evidence_values = (
        physical.height_in,
        physical.weight_lbs,
        physical.wingspan_in,
        physical.forty_time,
        speed_rating,
        accel_rating,
        route_rating,
        catch_rating,
        release,
        carrying,
        break_tackle,
        awareness,
        throw_power,
        throw_accuracy,
        throw_pressure,
        throw_on_run,
        play_action,
        break_sack,
        vision,
        strength,
        agility,
        change_dir,
        trucking,
        stiff_arm,
        juke,
        spin,
        stamina,
    )

    return PlayerIdentityChannels(
        speed=speed,
        mobility=mobility,
        qb_execution=qb_execution,
        route_separation=route_separation,
        catchpoint=catchpoint,
        rush_creation=rush_creation,
        runner_power=runner_power,
        open_field=open_field,
        ball_security=ball_security,
        evidence_fields=sum(value is not None for value in evidence_values),
    )
