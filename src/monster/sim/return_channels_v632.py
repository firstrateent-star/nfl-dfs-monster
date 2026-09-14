from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from monster.sim.chaos_ecology import (
    DEFAULT_CHAOS_ECOLOGY,
    ChaosEcology,
    ReturnEvent,
    ReturnKind,
    _returner,
)
from monster.sim.football_state import FootballState, mirror_field
from monster.sim.play_kernel import PassResult, PlayEvent
from monster.sim.reality_v63 import resolve_turnover_return_v63
from monster.sim.return_tail_v631 import sample_return_yards_v631
from monster.sim.special_teams_v13 import SpecialTeamsEvent, SpecialTeamsType


@dataclass(frozen=True)
class ReturnDistanceProfileV632:
    zero_rate: float
    mean: float
    sd: float
    p20: float
    p40: float
    p60: float
    p80: float


@dataclass(frozen=True)
class ChannelReturnPriorsV632:
    """Historical priors that separate return selection from live-return distance.

    No field in this object is a touchdown probability. A score can only occur when a sampled
    live-ball return distance traverses the remaining physical field.
    """

    fumble_recovery: ReturnDistanceProfileV632
    punt_touchback_rate: float
    punt_live_return_rate: float
    punt_live_return: ReturnDistanceProfileV632
    kickoff_touchback_rate: float
    kickoff_live_return_rate: float
    kickoff_live_return: ReturnDistanceProfileV632


DEFAULT_CHANNEL_PRIORS_V632 = ChannelReturnPriorsV632(
    fumble_recovery=ReturnDistanceProfileV632(0.76, 3.5, 12.0, 0.065, 0.016, 0.012, 0.004),
    punt_touchback_rate=0.077,
    punt_live_return_rate=0.466,
    punt_live_return=ReturnDistanceProfileV632(0.19, 10.2, 13.0, 0.094, 0.036, 0.019, 0.007),
    kickoff_touchback_rate=0.207,
    kickoff_live_return_rate=0.941,
    kickoff_live_return=ReturnDistanceProfileV632(0.006, 25.9, 10.5, 0.862, 0.042, 0.009, 0.005),
)


def _number(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(number) else number


def _profile_from_row(
    row: Mapping[str, object], prefix: str, default: ReturnDistanceProfileV632
) -> ReturnDistanceProfileV632:
    zero = float(np.clip(_number(row, f"{prefix}_zero_return_rate", default.zero_rate), 0.0, 0.995))
    p20 = float(np.clip(_number(row, f"{prefix}_20_plus_rate", default.p20), 0.0, 1.0 - zero))
    p40 = float(np.clip(_number(row, f"{prefix}_40_plus_rate", default.p40), 0.0, p20))
    p60 = float(np.clip(_number(row, f"{prefix}_60_plus_rate", default.p60), 0.0, p40))
    p80 = float(np.clip(_number(row, f"{prefix}_80_plus_rate", default.p80), 0.0, p60))
    return ReturnDistanceProfileV632(
        zero_rate=zero,
        mean=float(np.clip(_number(row, f"{prefix}_return_mean", default.mean), 0.0, 60.0)),
        sd=float(np.clip(_number(row, f"{prefix}_return_sd", default.sd), 0.5, 50.0)),
        p20=p20,
        p40=p40,
        p60=p60,
        p80=p80,
    )


def channel_priors_from_policy_row(
    row: Mapping[str, object] | None,
) -> ChannelReturnPriorsV632:
    if not row:
        return DEFAULT_CHANNEL_PRIORS_V632
    d = DEFAULT_CHANNEL_PRIORS_V632
    return ChannelReturnPriorsV632(
        fumble_recovery=_profile_from_row(row, "fumble_recovery", d.fumble_recovery),
        punt_touchback_rate=float(
            np.clip(_number(row, "punt_touchback_rate_explicit", d.punt_touchback_rate), 0.0, 0.40)
        ),
        punt_live_return_rate=float(
            np.clip(
                _number(row, "punt_live_return_rate_after_touchback", d.punt_live_return_rate),
                0.05,
                0.95,
            )
        ),
        punt_live_return=_profile_from_row(row, "punt_live", d.punt_live_return),
        kickoff_touchback_rate=float(
            np.clip(
                _number(row, "kickoff_touchback_rate_explicit", d.kickoff_touchback_rate),
                0.02,
                0.95,
            )
        ),
        kickoff_live_return_rate=float(
            np.clip(
                _number(
                    row,
                    "kickoff_live_return_rate_after_touchback",
                    d.kickoff_live_return_rate,
                ),
                0.25,
                1.0,
            )
        ),
        kickoff_live_return=_profile_from_row(row, "kickoff_live", d.kickoff_live_return),
    )


def _sample_profile(
    profile: ReturnDistanceProfileV632,
    *,
    return_skill: float,
    rng: np.random.Generator,
    maximum: float,
) -> float:
    return sample_return_yards_v631(
        mean=profile.mean,
        sd=profile.sd,
        zero_rate=profile.zero_rate,
        twenty_plus_rate=profile.p20,
        forty_plus_rate=profile.p40,
        sixty_plus_rate=profile.p60,
        eighty_plus_rate=profile.p80,
        return_skill=return_skill,
        rng=rng,
        maximum=maximum,
    )


def simulate_punt_v632(
    rng: np.random.Generator,
    *,
    punter_skill: float = 1.0,
    returner_id: str | None = None,
    punter_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    priors: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632,
) -> SpecialTeamsEvent:
    blocked = rng.random() < ecology.blocked_punt_rate
    gross = 0.0 if blocked else float(np.clip(rng.normal(45.0 * punter_skill, 6.0), 20.0, 70.0))
    touchback = not blocked and rng.random() < priors.punt_touchback_rate
    if blocked or touchback:
        return SpecialTeamsEvent(
            SpecialTeamsType.PUNT,
            kick_distance=gross,
            touchback=touchback,
            blocked=blocked,
            returner_id=returner_id,
            punter_id=punter_id,
        )

    muffed = rng.random() < ecology.punt_muff_rate
    kicking_recovery = muffed and rng.random() < ecology.punt_muff_kicking_recovery_rate
    live_return = False
    return_yards = 0.0
    if not muffed:
        live_return = rng.random() < priors.punt_live_return_rate
        if live_return:
            return_yards = _sample_profile(
                priors.punt_live_return,
                return_skill=return_skill,
                rng=rng,
                maximum=100.0,
            )
    return SpecialTeamsEvent(
        SpecialTeamsType.PUNT,
        kick_distance=gross,
        return_yards=return_yards,
        touchback=False,
        blocked=False,
        returner_id=returner_id,
        punter_id=punter_id,
        fair_catch=not muffed and not live_return,
        muffed=muffed,
        kicking_team_recovery=kicking_recovery,
    )


def simulate_kickoff_v632(
    rng: np.random.Generator,
    *,
    returner_id: str | None = None,
    kicker_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    priors: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632,
) -> SpecialTeamsEvent:
    touchback = rng.random() < priors.kickoff_touchback_rate
    if touchback:
        return SpecialTeamsEvent(
            SpecialTeamsType.KICKOFF,
            kick_distance=65.0,
            touchback=True,
            returner_id=returner_id,
            kicker_id=kicker_id,
        )

    landing = float(np.clip(20.0 * rng.beta(2.0, 3.0), 0.0, 20.0))
    muffed = rng.random() < ecology.kickoff_muff_rate
    kicking_recovery = muffed and rng.random() < ecology.kickoff_muff_kicking_recovery_rate
    live_return = False
    return_yards = 0.0
    if not muffed:
        live_return = rng.random() < priors.kickoff_live_return_rate
        if live_return:
            return_yards = _sample_profile(
                priors.kickoff_live_return,
                return_skill=return_skill,
                rng=rng,
                maximum=100.0 - landing,
            )
    return SpecialTeamsEvent(
        SpecialTeamsType.KICKOFF,
        kick_distance=65.0,
        return_yards=return_yards,
        touchback=False,
        returner_id=returner_id,
        kicker_id=kicker_id,
        return_start_yardline_100=landing,
        fair_catch=not muffed and not live_return,
        muffed=muffed,
        kicking_team_recovery=kicking_recovery,
    )


def resolve_turnover_return_v632(
    before: FootballState,
    event: PlayEvent,
    *,
    defense,
    rng: np.random.Generator,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    priors: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632,
) -> ReturnEvent:
    """Use validated v6.3 interception geometry and recovery-specific fumble geometry."""

    if event.pass_result == PassResult.INTERCEPTION:
        return resolve_turnover_return_v63(
            before,
            event,
            defense=defense,
            rng=rng,
            ecology=ecology,
        )
    if not event.turnover:
        raise ValueError("turnover return resolution requires a turnover event")

    spot = float(np.clip(before.yardline_100 + float(event.yards), 1.0, 99.0))
    defender = _returner(defense, event.primary_defender_id)
    returner_id = None if defender is None else defender.player_id
    return_skill = 1.0 if defender is None else float(defender.returning)
    start = mirror_field(spot)
    distance_to_goal = 100.0 - start
    yards = _sample_profile(
        priors.fumble_recovery,
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
        kind=ReturnKind.FUMBLE,
        original_offense_team_id=before.possession,
        return_team_id=before.defense,
        returner_id=returner_id,
        change_spot_yardline_100=spot,
        return_yards=float(yards),
        receiving_yardline_100=float(receiving),
        touchdown=touchdown,
    )