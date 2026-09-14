from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Any

import numpy as np

from monster.sim.chaos_ecology import ChaosEcology, ReturnEvent, ReturnKind, _returner
from monster.sim.football_state import FootballState, mirror_field
from monster.sim.play_kernel import PassResult, PlayEvent
from monster.sim.reality_v63 import (
    GameEnvironmentV63,
    active_game_environment,
    apply_chaos_environment_v63,
)


_BAND_AUTHORITIES = np.asarray([-0.18, -0.08, 0.18, 0.48, 0.78, 1.10], dtype=float)


def _monotone_survivors(
    zero_rate: float,
    p20: float,
    p40: float,
    p60: float,
    p80: float,
) -> tuple[float, float, float, float]:
    live = max(1.0 - float(np.clip(zero_rate, 0.0, 0.98)), 0.0)
    p20 = float(np.clip(p20, 0.0, live))
    p40 = float(np.clip(p40, 0.0, p20))
    p60 = float(np.clip(p60, 0.0, p40))
    p80 = float(np.clip(p80, 0.0, p60))
    return p20, p40, p60, p80


def _scale_survivors(
    zero_rate: float,
    rates: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    p20, p40, p60, p80 = rates
    scaled = (
        p20 * factor**0.34,
        p40 * factor**0.68,
        p60 * factor**0.88,
        p80 * factor**1.04,
    )
    return _monotone_survivors(zero_rate, *scaled)


def apply_chaos_environment_v631(
    ecology: ChaosEcology,
    environment: GameEnvironmentV63,
) -> ChaosEcology:
    """Extend v6.3 chaos state across the full historical return survivor curve."""

    base = apply_chaos_environment_v63(ecology, environment)
    factor = float(environment.chaos_factor)
    int_tail = _scale_survivors(
        base.interception_zero_return_rate,
        (
            ecology.interception_20_plus_rate,
            ecology.interception_40_plus_rate,
            ecology.interception_60_plus_rate,
            ecology.interception_80_plus_rate,
        ),
        factor,
    )
    fum_tail = _scale_survivors(
        base.fumble_zero_return_rate,
        (
            ecology.fumble_20_plus_rate,
            ecology.fumble_40_plus_rate,
            ecology.fumble_60_plus_rate,
            ecology.fumble_80_plus_rate,
        ),
        factor,
    )
    punt_tail = _scale_survivors(
        base.punt_zero_return_rate,
        (
            ecology.punt_20_plus_rate,
            ecology.punt_40_plus_rate,
            ecology.punt_60_plus_rate,
            ecology.punt_80_plus_rate,
        ),
        factor,
    )
    kick_tail = _scale_survivors(
        0.0,
        (
            ecology.kickoff_20_plus_rate,
            ecology.kickoff_40_plus_rate,
            ecology.kickoff_60_plus_rate,
            ecology.kickoff_80_plus_rate,
        ),
        factor,
    )
    return replace(
        base,
        interception_20_plus_rate=int_tail[0],
        interception_40_plus_rate=int_tail[1],
        interception_60_plus_rate=int_tail[2],
        interception_80_plus_rate=int_tail[3],
        fumble_20_plus_rate=fum_tail[0],
        fumble_40_plus_rate=fum_tail[1],
        fumble_60_plus_rate=fum_tail[2],
        fumble_80_plus_rate=fum_tail[3],
        punt_20_plus_rate=punt_tail[0],
        punt_40_plus_rate=punt_tail[1],
        punt_60_plus_rate=punt_tail[2],
        punt_80_plus_rate=punt_tail[3],
        kickoff_20_plus_rate=kick_tail[0],
        kickoff_40_plus_rate=kick_tail[1],
        kickoff_60_plus_rate=kick_tail[2],
        kickoff_80_plus_rate=kick_tail[3],
    )


def _band_probabilities(
    *,
    zero_rate: float,
    p20: float,
    p40: float,
    p60: float,
    p80: float,
    edge: float,
) -> np.ndarray:
    p20, p40, p60, p80 = _monotone_survivors(zero_rate, p20, p40, p60, p80)
    zero = float(np.clip(zero_rate, 0.0, 0.98))
    probabilities = np.asarray(
        [
            zero,
            max(1.0 - zero - p20, 0.0),
            max(p20 - p40, 0.0),
            max(p40 - p60, 0.0),
            max(p60 - p80, 0.0),
            max(p80, 0.0),
        ],
        dtype=float,
    )
    edge = float(np.clip(edge, 0.62, 1.62))
    probabilities *= np.power(edge, _BAND_AUTHORITIES)
    total = float(probabilities.sum())
    if total <= 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return probabilities / total


def _bounded_gamma_below(
    *,
    mean: float,
    sd: float,
    edge: float,
    upper: float,
    rng: np.random.Generator,
) -> float:
    live_mean = max(mean * edge, 0.5)
    live_sd = max(sd * sqrt(max(edge, 0.2)), 0.75)
    shape = max((live_mean / live_sd) ** 2, 0.25)
    scale = max((live_sd**2) / live_mean, 0.05)
    for _ in range(12):
        value = float(rng.gamma(shape, scale))
        if value < upper:
            return value
    return float(min(rng.uniform(0.0, upper), upper - 1e-6))


def sample_return_yards_v631(
    *,
    mean: float,
    sd: float,
    zero_rate: float,
    forty_plus_rate: float,
    return_skill: float,
    rng: np.random.Generator,
    maximum: float = 100.0,
    twenty_plus_rate: float | None = None,
    sixty_plus_rate: float | None = None,
    eighty_plus_rate: float | None = None,
) -> float:
    """Sample an NFL return from historical distance-survivor bands.

    The draw is a return *distance*, never a touchdown. If the field only requires 17 yards and
    the football return draw lands in the historical 20-39 band, the geometry naturally produces
    a score by clipping the return at the goal line. Conversely, an 80+ branch still cannot score
    when more field remains than the sampled return covers.
    """

    if maximum <= 0.0:
        return 0.0

    p40 = float(np.clip(forty_plus_rate, 0.0, 0.40))
    p20 = max(p40, p40 * 3.6) if twenty_plus_rate is None else float(twenty_plus_rate)
    p60 = p40 * 0.30 if sixty_plus_rate is None else float(sixty_plus_rate)
    p80 = p60 * 0.28 if eighty_plus_rate is None else float(eighty_plus_rate)

    environment = active_game_environment()
    lane_factor = 1.0
    if environment is not None:
        lane_factor = float(
            np.clip(
                environment.explosive_factor**0.16 * environment.chaos_factor**0.08,
                0.86,
                1.18,
            )
        )
    edge = float(np.clip(float(return_skill) * lane_factor, 0.62, 1.62))
    probabilities = _band_probabilities(
        zero_rate=zero_rate,
        p20=p20,
        p40=p40,
        p60=p60,
        p80=p80,
        edge=edge,
    )
    band = int(rng.choice(6, p=probabilities))
    if band == 0:
        raw = 0.0
    elif band == 1:
        raw = _bounded_gamma_below(mean=mean, sd=sd, edge=edge, upper=20.0, rng=rng)
    elif band == 2:
        raw = 20.0 + 19.999 * float(rng.beta(2.0, 2.0))
    elif band == 3:
        raw = 40.0 + 19.999 * float(rng.beta(1.8, 2.0))
    elif band == 4:
        raw = 60.0 + 19.999 * float(rng.beta(1.6, 2.0))
    else:
        raw = 80.0 + float(rng.gamma(1.25, max(6.5 * edge, 2.0)))
    return float(np.clip(raw, 0.0, maximum))


def resolve_turnover_return_v631(
    before: FootballState,
    event: PlayEvent,
    *,
    defense: Any,
    rng: np.random.Generator,
    ecology: ChaosEcology,
) -> ReturnEvent:
    """Resolve turnover returns from absolute historical return capability + field geometry."""

    if not event.turnover:
        raise ValueError("turnover return resolution requires a turnover event")

    interception = event.pass_result == PassResult.INTERCEPTION
    kind = ReturnKind.INTERCEPTION if interception else ReturnKind.FUMBLE
    if interception:
        raw_spot = before.yardline_100 + float(event.air_yards)
        if raw_spot >= 100.0:
            return ReturnEvent(
                kind=kind,
                original_offense_team_id=before.possession,
                return_team_id=before.defense,
                returner_id=event.primary_defender_id,
                change_spot_yardline_100=100.0,
                return_yards=0.0,
                receiving_yardline_100=20.0,
                touchdown=False,
                touchback=True,
            )
        spot = float(np.clip(raw_spot, 1.0, 99.0))
        zero_rate = ecology.interception_zero_return_rate
        mean = ecology.interception_return_mean
        sd = ecology.interception_return_sd
        p20 = ecology.interception_20_plus_rate
        p40 = ecology.interception_40_plus_rate
        p60 = ecology.interception_60_plus_rate
        p80 = ecology.interception_80_plus_rate
    else:
        spot = float(np.clip(before.yardline_100 + float(event.yards), 1.0, 99.0))
        zero_rate = ecology.fumble_zero_return_rate
        mean = ecology.fumble_return_mean
        sd = ecology.fumble_return_sd
        p20 = ecology.fumble_20_plus_rate
        p40 = ecology.fumble_40_plus_rate
        p60 = ecology.fumble_60_plus_rate
        p80 = ecology.fumble_80_plus_rate

    defender = _returner(defense, event.primary_defender_id)
    returner_id = None if defender is None else defender.player_id
    return_skill = 1.0 if defender is None else float(defender.returning)
    start = mirror_field(spot)
    distance_to_goal = 100.0 - start
    yards = sample_return_yards_v631(
        mean=mean,
        sd=sd,
        zero_rate=zero_rate,
        twenty_plus_rate=p20,
        forty_plus_rate=p40,
        sixty_plus_rate=p60,
        eighty_plus_rate=p80,
        return_skill=return_skill,
        rng=rng,
        maximum=max(distance_to_goal, 0.0),
    )
    receiving = start + yards
    touchdown = receiving >= 100.0 - 1e-9
    if touchdown:
        receiving = 100.0
        yards = distance_to_goal
    return ReturnEvent(
        kind=kind,
        original_offense_team_id=before.possession,
        return_team_id=before.defense,
        returner_id=returner_id,
        change_spot_yardline_100=spot,
        return_yards=float(yards),
        receiving_yardline_100=float(receiving),
        touchdown=touchdown,
    )