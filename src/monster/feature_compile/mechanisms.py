from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import tanh

import numpy as np

from monster.feature_compile.madden_attributes import MaddenAttributeVector, channel_mean
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


@dataclass(frozen=True)
class PlayerMechanismInputs:
    """Raw non-market inputs used only where they have football jurisdiction."""

    height_in: float | None = None
    weight_lbs: float | None = None
    wingspan_in: float | None = None
    forty_time: float | None = None
    madden_speed: float | None = None
    madden_acceleration: float | None = None
    madden_route_running: float | None = None
    madden_catching: float | None = None
    madden_attributes: MaddenAttributeVector = field(default_factory=MaddenAttributeVector)
    age_years: float | None = None
    career_workload: float | None = None
    unit_continuity: float | None = None
    active_probability: float | None = None
    effectiveness_if_active: float | None = None
    astrology_shadow_signal: float | None = None


@dataclass(frozen=True)
class TeamMechanismInputs:
    """Team/environment inputs compiled before Monte Carlo worlds are generated."""

    team_madden_ovr: float | None = None
    offensive_line_index: float | None = None
    opponent_front_index: float | None = None
    unit_continuity: float | None = None
    coach_policy_entropy: float | None = None
    wind_mph: float | None = None
    precipitation_probability: float | None = None
    temperature_f: float | None = None
    dome: bool = False


@dataclass(frozen=True)
class PlayerMechanismTrace:
    speed_signal: float
    catchpoint_signal: float
    rushing_signal: float
    passing_signal: float
    ball_security_signal: float
    madden_attribute_count: int
    biology_uncertainty: float
    continuity_signal: float
    astrology_shadow_signal: float | None


@dataclass(frozen=True)
class TeamMechanismTrace:
    personnel_signal: float
    weather_effect: float
    continuity_signal: float
    coaching_entropy: float


_POSITION_REFERENCE = {
    "QB": {"height": 75.0, "weight": 225.0, "forty": 4.80, "wingspan": 79.0, "workload": 4500.0},
    "RB": {"height": 70.0, "weight": 210.0, "forty": 4.50, "wingspan": 76.0, "workload": 1100.0},
    "WR": {"height": 73.0, "weight": 200.0, "forty": 4.48, "wingspan": 77.0, "workload": 700.0},
    "TE": {"height": 77.0, "weight": 245.0, "forty": 4.70, "wingspan": 80.0, "workload": 650.0},
}

_AGE_UNCERTAINTY_START = {"QB": 34.0, "RB": 28.0, "WR": 30.0, "TE": 31.0}


def _signal(value: float | None, center: float, scale: float, *, reverse: bool = False) -> float:
    if value is None:
        return 0.0
    raw = (float(value) - center) / max(scale, 1e-6)
    if reverse:
        raw *= -1.0
    return float(tanh(raw))


def _weighted_signal(parts: tuple[tuple[float, float], ...]) -> float:
    weight = sum(abs(w) for _, w in parts if w != 0.0)
    if weight <= 0:
        return 0.0
    return float(np.clip(sum(value * w for value, w in parts) / weight, -1.0, 1.0))


def _position_reference(position: str) -> dict[str, float]:
    return _POSITION_REFERENCE.get(position.upper(), _POSITION_REFERENCE["WR"])


def _madden_signal(inputs: PlayerMechanismInputs, channel: str) -> float:
    return _signal(channel_mean(inputs.madden_attributes, channel), 82.0, 10.0)


def compile_player_mechanisms(
    base: PlayerState, inputs: PlayerMechanismInputs
) -> tuple[PlayerState, PlayerMechanismTrace]:
    """Compile rich player evidence into bounded mechanism-specific modifiers.

    Every available numeric Madden attribute is preserved in ``madden_attributes``.
    Only semantically relevant channels receive authority here, so the same rating is
    not a generic projection multiplier and Madden never awards fantasy points directly.
    """
    ref = _position_reference(base.position)
    forty_signal = _signal(inputs.forty_time, ref["forty"], 0.16, reverse=True)
    speed_rating = inputs.madden_speed or inputs.madden_attributes.get("speed")
    accel_rating = inputs.madden_acceleration or inputs.madden_attributes.get("acceleration")
    route_rating = inputs.madden_route_running or channel_mean(
        inputs.madden_attributes, "receiver_release"
    )
    catch_rating = inputs.madden_catching or channel_mean(inputs.madden_attributes, "catching")
    madden_speed_signal = _signal(speed_rating, 85.0, 8.0)
    acceleration_signal = _signal(accel_rating, 85.0, 8.0)
    route_signal = _signal(route_rating, 82.0, 10.0)
    catching_signal = _signal(catch_rating, 82.0, 10.0)
    height_signal = _signal(inputs.height_in, ref["height"], 3.0)
    weight_signal = _signal(inputs.weight_lbs, ref["weight"], 24.0)
    wingspan_signal = _signal(inputs.wingspan_in, ref["wingspan"], 4.0)

    movement = _madden_signal(inputs, "movement")
    open_field = _madden_signal(inputs, "open_field_rushing")
    passing = _weighted_signal(
        (
            (_madden_signal(inputs, "qb_accuracy"), 0.75),
            (_madden_signal(inputs, "qb_arm"), 0.25),
        )
    )
    ball_security = _madden_signal(inputs, "ball_security")

    speed_signal = _weighted_signal(
        (
            (forty_signal, 0.45),
            (madden_speed_signal, 0.25),
            (acceleration_signal, 0.15),
            (movement, 0.15),
        )
    )
    catchpoint_signal = _weighted_signal(
        (
            (height_signal, 0.20),
            (wingspan_signal, 0.20),
            (catching_signal, 0.35),
            (route_signal, 0.25),
        )
    )
    rushing_signal = _weighted_signal(
        (
            (forty_signal, 0.20),
            (madden_speed_signal, 0.15),
            (acceleration_signal, 0.15),
            (weight_signal, 0.20),
            (open_field, 0.30),
        )
    )

    explosive_modifier = float(
        np.clip(base.explosive_modifier * (1.0 + 0.05 * speed_signal), 0.94, 1.06)
    )
    catchpoint_modifier = float(
        np.clip(base.catchpoint_modifier * (1.0 + 0.05 * catchpoint_signal), 0.94, 1.06)
    )
    rushing_modifier = float(
        np.clip(base.rushing_efficiency_modifier * (1.0 + 0.05 * rushing_signal), 0.94, 1.06)
    )

    biology_uncertainty = 0.0
    age_start = _AGE_UNCERTAINTY_START.get(base.position.upper(), 30.0)
    if inputs.age_years is not None and inputs.career_workload is not None:
        age_excess = max(float(inputs.age_years) - age_start, 0.0) / 5.0
        workload_signal = max(
            _signal(inputs.career_workload, ref["workload"], ref["workload"] * 0.55), 0.0
        )
        biology_uncertainty = float(
            np.clip(0.025 * age_excess * workload_signal, 0.0, 0.05)
        )

    continuity_signal = 0.0
    if inputs.unit_continuity is not None:
        continuity_signal = float(
            np.clip((float(inputs.unit_continuity) - 0.5) * 2.0, -1.0, 1.0)
        )
    role_uncertainty = float(
        np.clip(
            base.role_uncertainty * (1.0 - 0.20 * continuity_signal) + biology_uncertainty,
            0.025,
            0.35,
        )
    )
    active_probability = (
        base.active_probability
        if inputs.active_probability is None
        else float(np.clip(inputs.active_probability, 0.0, 1.0))
    )
    effectiveness = (
        base.effectiveness_if_active
        if inputs.effectiveness_if_active is None
        else float(np.clip(inputs.effectiveness_if_active, 0.35, 1.10))
    )

    compiled = replace(
        base,
        active_probability=active_probability,
        effectiveness_if_active=effectiveness,
        role_uncertainty=role_uncertainty,
        explosive_modifier=explosive_modifier,
        catchpoint_modifier=catchpoint_modifier,
        rushing_efficiency_modifier=rushing_modifier,
    )
    trace = PlayerMechanismTrace(
        speed_signal=speed_signal,
        catchpoint_signal=catchpoint_signal,
        rushing_signal=rushing_signal,
        passing_signal=passing,
        ball_security_signal=ball_security,
        madden_attribute_count=len(inputs.madden_attributes.values),
        biology_uncertainty=biology_uncertainty,
        continuity_signal=continuity_signal,
        astrology_shadow_signal=inputs.astrology_shadow_signal,
    )
    return compiled, trace


def _weather_effect(inputs: TeamMechanismInputs) -> float:
    if inputs.dome:
        return 0.0
    wind_penalty = (
        0.0
        if inputs.wind_mph is None
        else -0.0030 * max(float(inputs.wind_mph) - 10.0, 0.0)
    )
    precip = (
        0.0
        if inputs.precipitation_probability is None
        else float(np.clip(inputs.precipitation_probability, 0.0, 1.0))
    )
    precip_penalty = -0.018 * precip
    temperature_penalty = 0.0
    if inputs.temperature_f is not None:
        temperature = float(inputs.temperature_f)
        if temperature < 25.0:
            temperature_penalty = -0.010 * min((25.0 - temperature) / 20.0, 1.0)
        elif temperature > 95.0:
            temperature_penalty = -0.006 * min((temperature - 95.0) / 15.0, 1.0)
    return float(np.clip(wind_penalty + precip_penalty + temperature_penalty, -0.06, 0.0))


def compile_team_mechanisms(
    team: TeamState, pool: TeamPlayerPool, inputs: TeamMechanismInputs
) -> tuple[TeamState, TeamPlayerPool, TeamMechanismTrace]:
    """Compile team personnel, continuity, coaching and environment into bounded effects."""
    madden_signal = _signal(inputs.team_madden_ovr, 82.0, 7.0)
    line_signal = _signal(inputs.offensive_line_index, 0.0, 2.5)
    front_signal = _signal(inputs.opponent_front_index, 0.0, 2.5, reverse=True)
    personnel_signal = _weighted_signal(
        ((madden_signal, 0.30), (line_signal, 0.40), (front_signal, 0.30))
    )
    personnel_effect = float(np.clip(0.035 * personnel_signal, -0.035, 0.035))
    continuity = team.continuity
    continuity_signal = 0.0
    if inputs.unit_continuity is not None:
        continuity = float(np.clip(inputs.unit_continuity, 0.0, 1.0))
        continuity_signal = float(np.clip((continuity - 0.5) * 2.0, -1.0, 1.0))
    coaching_entropy = team.coaching_entropy
    if inputs.coach_policy_entropy is not None:
        coaching_entropy = float(np.clip(inputs.coach_policy_entropy, 0.02, 0.35))
    weather_effect = _weather_effect(inputs)
    compiled_team = replace(
        team,
        continuity=continuity,
        coaching_entropy=coaching_entropy,
        weather_effect=weather_effect,
        physical_madden_effect=personnel_effect,
    )
    pass_weather_shift = float(np.clip(weather_effect * 0.35, -0.025, 0.0))
    compiled_pool = replace(
        pool,
        neutral_pass_rate=float(
            np.clip(pool.neutral_pass_rate + pass_weather_shift, 0.34, 0.72)
        ),
        targetable_dropback_rate=float(
            np.clip(pool.targetable_dropback_rate + weather_effect * 0.20, 0.84, 0.98)
        ),
        play_volume_uncertainty=float(
            np.clip(
                pool.play_volume_uncertainty
                + 0.03 * coaching_entropy
                - 0.015 * continuity_signal,
                0.025,
                0.18,
            )
        ),
    )
    trace = TeamMechanismTrace(
        personnel_signal=personnel_signal,
        weather_effect=weather_effect,
        continuity_signal=continuity_signal,
        coaching_entropy=coaching_entropy,
    )
    return compiled_team, compiled_pool, trace
