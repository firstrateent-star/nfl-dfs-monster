from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any

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
from monster.sim.reality_v63 import active_game_environment, resolve_turnover_return_v63
from monster.sim.return_channels_v632 import (
    ChannelReturnPriorsV632,
    DEFAULT_CHANNEL_PRIORS_V632,
)
from monster.sim.special_teams_v13 import SpecialTeamsEvent, SpecialTeamsType


_BUCKET_NAMES = ("00_20", "20_40", "40_60", "60_80", "80_100")
_BAND_AUTHORITIES = np.asarray([-0.22, -0.14, -0.06, 0.02, 0.12, 0.32, 0.58, 0.82, 1.08])


@dataclass(frozen=True)
class GeometryReturnProfileV633:
    rows: int
    zero_rate: float
    mean: float
    sd: float
    p5: float
    p10: float
    p15: float
    p20: float
    p40: float
    p60: float
    p80: float
    upper80_rows: int
    upper80_mean: float
    upper80_sd: float


@dataclass(frozen=True)
class ReturnGeometryPriorsV633:
    fumble_global: GeometryReturnProfileV633
    fumble_buckets: Mapping[str, GeometryReturnProfileV633]
    punt_global: GeometryReturnProfileV633
    punt_buckets: Mapping[str, GeometryReturnProfileV633]
    kickoff_global: GeometryReturnProfileV633
    kickoff_landing_mean: float
    kickoff_landing_sd: float
    kickoff_landing_rows: int
    kickoff_returned_drive_start_mean: float


DEFAULT_FUMBLE_PROFILE_V633 = GeometryReturnProfileV633(
    rows=245,
    zero_rate=0.763,
    mean=3.46,
    sd=12.0,
    p5=0.15,
    p10=0.11,
    p15=0.08,
    p20=0.065,
    p40=0.016,
    p60=0.012,
    p80=0.004,
    upper80_rows=1,
    upper80_mean=86.0,
    upper80_sd=5.0,
)
DEFAULT_PUNT_PROFILE_V633 = GeometryReturnProfileV633(
    rows=827,
    zero_rate=0.191,
    mean=10.22,
    sd=13.0,
    p5=0.53,
    p10=0.31,
    p15=0.18,
    p20=0.094,
    p40=0.036,
    p60=0.019,
    p80=0.007,
    upper80_rows=6,
    upper80_mean=90.5,
    upper80_sd=6.0,
)
DEFAULT_KICKOFF_PROFILE_V633 = GeometryReturnProfileV633(
    rows=2078,
    zero_rate=0.006,
    mean=25.92,
    sd=10.5,
    p5=0.99,
    p10=0.98,
    p15=0.95,
    p20=0.862,
    p40=0.042,
    p60=0.009,
    p80=0.0048,
    upper80_rows=10,
    upper80_mean=90.0,
    upper80_sd=7.0,
)
DEFAULT_RETURN_GEOMETRY_V633 = ReturnGeometryPriorsV633(
    fumble_global=DEFAULT_FUMBLE_PROFILE_V633,
    fumble_buckets={name: DEFAULT_FUMBLE_PROFILE_V633 for name in _BUCKET_NAMES},
    punt_global=DEFAULT_PUNT_PROFILE_V633,
    punt_buckets={name: DEFAULT_PUNT_PROFILE_V633 for name in _BUCKET_NAMES},
    kickoff_global=DEFAULT_KICKOFF_PROFILE_V633,
    kickoff_landing_mean=4.2,
    kickoff_landing_sd=3.5,
    kickoff_landing_rows=2000,
    kickoff_returned_drive_start_mean=29.6,
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


def _integer(row: Mapping[str, object], key: str, default: int) -> int:
    return max(int(round(_number(row, key, float(default)))), 0)


def _monotone(profile: GeometryReturnProfileV633) -> GeometryReturnProfileV633:
    live = max(1.0 - float(np.clip(profile.zero_rate, 0.0, 0.995)), 0.0)
    p5 = float(np.clip(profile.p5, 0.0, live))
    p10 = float(np.clip(profile.p10, 0.0, p5))
    p15 = float(np.clip(profile.p15, 0.0, p10))
    p20 = float(np.clip(profile.p20, 0.0, p15))
    p40 = float(np.clip(profile.p40, 0.0, p20))
    p60 = float(np.clip(profile.p60, 0.0, p40))
    p80 = float(np.clip(profile.p80, 0.0, p60))
    return replace(
        profile,
        zero_rate=float(np.clip(profile.zero_rate, 0.0, 0.995)),
        p5=p5,
        p10=p10,
        p15=p15,
        p20=p20,
        p40=p40,
        p60=p60,
        p80=p80,
    )


def _profile_from_row(
    row: Mapping[str, object], prefix: str, default: GeometryReturnProfileV633
) -> GeometryReturnProfileV633:
    profile = GeometryReturnProfileV633(
        rows=_integer(row, f"{prefix}_rows", default.rows),
        zero_rate=_number(row, f"{prefix}_zero_rate", default.zero_rate),
        mean=_number(row, f"{prefix}_mean", default.mean),
        sd=max(_number(row, f"{prefix}_sd", default.sd), 0.5),
        p5=_number(row, f"{prefix}_p5", default.p5),
        p10=_number(row, f"{prefix}_p10", default.p10),
        p15=_number(row, f"{prefix}_p15", default.p15),
        p20=_number(row, f"{prefix}_p20", default.p20),
        p40=_number(row, f"{prefix}_p40", default.p40),
        p60=_number(row, f"{prefix}_p60", default.p60),
        p80=_number(row, f"{prefix}_p80", default.p80),
        upper80_rows=_integer(row, f"{prefix}_upper80_rows", default.upper80_rows),
        upper80_mean=_number(row, f"{prefix}_upper80_mean", default.upper80_mean),
        upper80_sd=max(_number(row, f"{prefix}_upper80_sd", default.upper80_sd), 1.0),
    )
    return _monotone(profile)


def geometry_priors_from_policy_row(
    row: Mapping[str, object] | None,
) -> ReturnGeometryPriorsV633:
    if not row:
        return DEFAULT_RETURN_GEOMETRY_V633
    fumble_global = _profile_from_row(row, "v633_fumble_global", DEFAULT_FUMBLE_PROFILE_V633)
    punt_global = _profile_from_row(row, "v633_punt_global", DEFAULT_PUNT_PROFILE_V633)
    kickoff_global = _profile_from_row(row, "v633_kickoff_global", DEFAULT_KICKOFF_PROFILE_V633)
    return ReturnGeometryPriorsV633(
        fumble_global=fumble_global,
        fumble_buckets={
            name: _profile_from_row(row, f"v633_fumble_{name}", fumble_global)
            for name in _BUCKET_NAMES
        },
        punt_global=punt_global,
        punt_buckets={
            name: _profile_from_row(row, f"v633_punt_{name}", punt_global)
            for name in _BUCKET_NAMES
        },
        kickoff_global=kickoff_global,
        kickoff_landing_mean=float(
            np.clip(_number(row, "v633_kickoff_landing_mean", 4.2), 0.2, 15.0)
        ),
        kickoff_landing_sd=float(
            np.clip(_number(row, "v633_kickoff_landing_sd", 3.5), 0.5, 12.0)
        ),
        kickoff_landing_rows=_integer(row, "v633_kickoff_landing_rows", 0),
        kickoff_returned_drive_start_mean=float(
            np.clip(_number(row, "v633_kickoff_returned_drive_start_mean", 29.6), 20.0, 40.0)
        ),
    )


def required_distance_bucket(required_distance: float) -> str:
    value = float(np.clip(required_distance, 0.0, 100.0))
    if value <= 20.0:
        return "00_20"
    if value <= 40.0:
        return "20_40"
    if value <= 60.0:
        return "40_60"
    if value <= 80.0:
        return "60_80"
    return "80_100"


def _blend_profile(
    local: GeometryReturnProfileV633,
    global_profile: GeometryReturnProfileV633,
    *,
    shrinkage_rows: float = 35.0,
) -> GeometryReturnProfileV633:
    weight = float(local.rows / max(local.rows + shrinkage_rows, 1.0))

    def blend(a: float, b: float) -> float:
        return weight * a + (1.0 - weight) * b

    upper_weight = float(local.upper80_rows / max(local.upper80_rows + 8.0, 1.0))
    profile = GeometryReturnProfileV633(
        rows=local.rows,
        zero_rate=blend(local.zero_rate, global_profile.zero_rate),
        mean=blend(local.mean, global_profile.mean),
        sd=max(blend(local.sd, global_profile.sd), 0.5),
        p5=blend(local.p5, global_profile.p5),
        p10=blend(local.p10, global_profile.p10),
        p15=blend(local.p15, global_profile.p15),
        p20=blend(local.p20, global_profile.p20),
        p40=blend(local.p40, global_profile.p40),
        p60=blend(local.p60, global_profile.p60),
        p80=blend(local.p80, global_profile.p80),
        upper80_rows=local.upper80_rows,
        upper80_mean=(
            upper_weight * local.upper80_mean
            + (1.0 - upper_weight) * global_profile.upper80_mean
        ),
        upper80_sd=max(
            upper_weight * local.upper80_sd
            + (1.0 - upper_weight) * global_profile.upper80_sd,
            1.0,
        ),
    )
    return _monotone(profile)


def profile_for_required_distance(
    priors: ReturnGeometryPriorsV633,
    *,
    channel: str,
    required_distance: float,
) -> GeometryReturnProfileV633:
    bucket = required_distance_bucket(required_distance)
    if channel == "fumble":
        return _blend_profile(priors.fumble_buckets[bucket], priors.fumble_global)
    if channel == "punt":
        return _blend_profile(priors.punt_buckets[bucket], priors.punt_global)
    raise ValueError(f"unsupported geometry channel: {channel}")


def _band_probabilities(profile: GeometryReturnProfileV633, edge: float) -> np.ndarray:
    p = _monotone(profile)
    live = max(1.0 - p.zero_rate, 0.0)
    probabilities = np.asarray(
        [
            p.zero_rate,
            max(live - p.p5, 0.0),
            max(p.p5 - p.p10, 0.0),
            max(p.p10 - p.p15, 0.0),
            max(p.p15 - p.p20, 0.0),
            max(p.p20 - p.p40, 0.0),
            max(p.p40 - p.p60, 0.0),
            max(p.p60 - p.p80, 0.0),
            max(p.p80, 0.0),
        ],
        dtype=float,
    )
    adjusted_edge = float(np.clip(edge, 0.65, 1.55))
    probabilities *= np.power(adjusted_edge, _BAND_AUTHORITIES)
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities[0] = 1.0
        return probabilities
    return probabilities / total


def _tail_edge(return_skill: float) -> float:
    environment = active_game_environment()
    environment_factor = 1.0
    if environment is not None:
        environment_factor = float(
            np.clip(
                environment.explosive_factor**0.14 * environment.chaos_factor**0.08,
                0.87,
                1.16,
            )
        )
    return float(np.clip(return_skill * environment_factor, 0.65, 1.55))


def sample_geometry_return_distance_v633(
    profile: GeometryReturnProfileV633,
    *,
    return_skill: float,
    rng: np.random.Generator,
    maximum: float,
) -> float:
    """Sample a physical return distance from empirical survivor bands, never a TD state."""

    if maximum <= 0.0:
        return 0.0
    edge = _tail_edge(return_skill)
    band = int(rng.choice(9, p=_band_probabilities(profile, edge)))
    if band == 0:
        raw = 0.0
    elif band == 1:
        raw = 4.999 * float(rng.beta(1.7, 1.8))
    elif band == 2:
        raw = 5.0 + 4.999 * float(rng.beta(1.8, 1.8))
    elif band == 3:
        raw = 10.0 + 4.999 * float(rng.beta(1.8, 1.8))
    elif band == 4:
        raw = 15.0 + 4.999 * float(rng.beta(1.8, 1.8))
    elif band == 5:
        raw = 20.0 + 19.999 * float(rng.beta(2.0, 2.0))
    elif band == 6:
        raw = 40.0 + 19.999 * float(rng.beta(1.8, 2.0))
    elif band == 7:
        raw = 60.0 + 19.999 * float(rng.beta(1.6, 2.0))
    else:
        excess_mean = max((profile.upper80_mean - 80.0) * edge, 1.0)
        excess_sd = max(profile.upper80_sd * sqrt(edge), 1.0)
        shape = max((excess_mean / excess_sd) ** 2, 0.30)
        scale = max((excess_sd**2) / excess_mean, 0.20)
        raw = 80.0 + float(rng.gamma(shape, scale))
    return float(np.clip(raw, 0.0, maximum))


def sample_kickoff_landing_v633(
    priors: ReturnGeometryPriorsV633,
    rng: np.random.Generator,
) -> float:
    mean = max(priors.kickoff_landing_mean, 0.2)
    sd = max(priors.kickoff_landing_sd, 0.5)
    shape = max((mean / sd) ** 2, 0.20)
    scale = max((sd**2) / mean, 0.05)
    for _ in range(12):
        value = float(rng.gamma(shape, scale))
        if value <= 20.0:
            return max(value, 0.0)
    return float(np.clip(rng.normal(mean, sd), 0.0, 20.0))


def simulate_punt_selection_v633(
    rng: np.random.Generator,
    *,
    punter_skill: float = 1.0,
    returner_id: str | None = None,
    punter_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    channels: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632,
) -> SpecialTeamsEvent:
    del return_skill
    blocked = rng.random() < ecology.blocked_punt_rate
    gross = 0.0 if blocked else float(np.clip(rng.normal(45.0 * punter_skill, 6.0), 20.0, 70.0))
    touchback = not blocked and rng.random() < channels.punt_touchback_rate
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
    live_return = False if muffed else rng.random() < channels.punt_live_return_rate
    return SpecialTeamsEvent(
        SpecialTeamsType.PUNT,
        kick_distance=gross,
        return_yards=0.0,
        touchback=False,
        blocked=False,
        returner_id=returner_id,
        punter_id=punter_id,
        fair_catch=not muffed and not live_return,
        muffed=muffed,
        kicking_team_recovery=kicking_recovery,
    )


def punt_with_geometry_distance_v633(
    before: FootballState,
    punt: SpecialTeamsEvent,
    *,
    rng: np.random.Generator,
    priors: ReturnGeometryPriorsV633,
) -> SpecialTeamsEvent:
    if punt.blocked or punt.touchback or punt.muffed or punt.fair_catch:
        return punt
    physical_end = float(
        np.clip(before.yardline_100 + max(float(punt.kick_distance), 0.0), 1.0, 99.0)
    )
    profile = profile_for_required_distance(
        priors,
        channel="punt",
        required_distance=physical_end,
    )
    yards = sample_geometry_return_distance_v633(
        profile,
        return_skill=1.0,
        rng=rng,
        maximum=physical_end,
    )
    return replace(punt, return_yards=yards)


def simulate_kickoff_v633(
    rng: np.random.Generator,
    *,
    returner_id: str | None = None,
    kicker_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    channels: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632,
    geometry: ReturnGeometryPriorsV633 = DEFAULT_RETURN_GEOMETRY_V633,
) -> SpecialTeamsEvent:
    touchback = rng.random() < channels.kickoff_touchback_rate
    if touchback:
        return SpecialTeamsEvent(
            SpecialTeamsType.KICKOFF,
            kick_distance=65.0,
            touchback=True,
            returner_id=returner_id,
            kicker_id=kicker_id,
        )
    landing = sample_kickoff_landing_v633(geometry, rng)
    muffed = rng.random() < ecology.kickoff_muff_rate
    kicking_recovery = muffed and rng.random() < ecology.kickoff_muff_kicking_recovery_rate
    live_return = False if muffed else rng.random() < channels.kickoff_live_return_rate
    return_yards = 0.0
    if live_return:
        return_yards = sample_geometry_return_distance_v633(
            geometry.kickoff_global,
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


def resolve_turnover_return_v633(
    before: FootballState,
    event: PlayEvent,
    *,
    defense: Any,
    rng: np.random.Generator,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
    priors: ReturnGeometryPriorsV633 = DEFAULT_RETURN_GEOMETRY_V633,
) -> ReturnEvent:
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
    profile = profile_for_required_distance(
        priors,
        channel="fumble",
        required_distance=distance_to_goal,
    )
    yards = sample_geometry_return_distance_v633(
        profile,
        return_skill=return_skill,
        rng=rng,
        maximum=distance_to_goal,
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
