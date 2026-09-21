from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY, ChaosEcology
from monster.sim.return_channels_v632 import (
    DEFAULT_CHANNEL_PRIORS_V632,
    ChannelReturnPriorsV632,
    channel_priors_from_policy_row,
)
from monster.sim.return_geometry_v633 import (
    DEFAULT_RETURN_GEOMETRY_V633,
    ReturnGeometryPriorsV633,
    geometry_priors_from_policy_row,
    resolve_turnover_return_v633,
    simulate_kickoff_v633,
)
from monster.sim.special_teams_v13 import SpecialTeamsEvent, SpecialTeamsType


@dataclass(frozen=True)
class FieldGoalBucketV725:
    low: float
    high: float | None
    make_rate_unblocked: float
    attempts: int
    unblocked_attempts: int


@dataclass(frozen=True)
class FieldGoalEcologyV725:
    buckets: tuple[FieldGoalBucketV725, ...]


_DEFAULT_FIELD_GOALS = FieldGoalEcologyV725(
    buckets=(
        FieldGoalBucketV725(0.0, 29.0, 0.985, 0, 0),
        FieldGoalBucketV725(30.0, 39.0, 0.955, 0, 0),
        FieldGoalBucketV725(40.0, 49.0, 0.885, 0, 0),
        FieldGoalBucketV725(50.0, 59.0, 0.735, 0, 0),
        FieldGoalBucketV725(60.0, None, 0.480, 0, 0),
    )
)

_ACTIVE_FIELD_GOALS = _DEFAULT_FIELD_GOALS
_ACTIVE_CHANNEL_PRIORS: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632
_ACTIVE_RETURN_GEOMETRY: ReturnGeometryPriorsV633 = DEFAULT_RETURN_GEOMETRY_V633
_ROWS: list[dict[str, object]] = []


def _number(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(out) else out


def _integer(row: Mapping[str, object], key: str, default: int = 0) -> int:
    return max(round(_number(row, key, float(default))), 0)


def field_goal_ecology_from_policy_row(
    row: Mapping[str, object] | None,
) -> FieldGoalEcologyV725:
    if not row:
        return _DEFAULT_FIELD_GOALS
    defaults = _DEFAULT_FIELD_GOALS.buckets
    specs = (
        ("00_29", defaults[0]),
        ("30_39", defaults[1]),
        ("40_49", defaults[2]),
        ("50_59", defaults[3]),
        ("60_plus", defaults[4]),
    )
    buckets = []
    for name, default in specs:
        buckets.append(
            FieldGoalBucketV725(
                low=default.low,
                high=default.high,
                make_rate_unblocked=float(
                    np.clip(
                        _number(
                            row,
                            f"v725_fg_{name}_make_rate_unblocked",
                            default.make_rate_unblocked,
                        ),
                        0.05,
                        0.995,
                    )
                ),
                attempts=_integer(row, f"v725_fg_{name}_attempts"),
                unblocked_attempts=_integer(
                    row, f"v725_fg_{name}_unblocked_attempts"
                ),
            )
        )
    return FieldGoalEcologyV725(tuple(buckets))


def configure_possession_ecology_v725(
    row: Mapping[str, object] | None,
) -> None:
    global _ACTIVE_FIELD_GOALS, _ACTIVE_CHANNEL_PRIORS, _ACTIVE_RETURN_GEOMETRY
    _ACTIVE_FIELD_GOALS = field_goal_ecology_from_policy_row(row)
    _ACTIVE_CHANNEL_PRIORS = channel_priors_from_policy_row(row)
    _ACTIVE_RETURN_GEOMETRY = geometry_priors_from_policy_row(row)


def active_field_goal_ecology_v725() -> FieldGoalEcologyV725:
    return _ACTIVE_FIELD_GOALS


def _field_goal_bucket(distance: float) -> FieldGoalBucketV725:
    value = float(distance)
    for bucket in _ACTIVE_FIELD_GOALS.buckets:
        if value < bucket.low:
            continue
        if bucket.high is None or value <= bucket.high:
            return bucket
    return _ACTIVE_FIELD_GOALS.buckets[-1]


def simulate_field_goal_v725(
    rng: np.random.Generator,
    *,
    distance: float,
    kicking_skill: float = 1.0,
    kicker_id: str | None = None,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> SpecialTeamsEvent:
    """Resolve a kick from leakage-safe distance evidence plus current kicker skill.

    Historical make probability is conditional on the kick not being blocked. The blocked
    branch stays separate and event-derived, so no score probability is ever injected.
    """
    bucket = _field_goal_bucket(distance)
    blocked = rng.random() < ecology.blocked_field_goal_rate
    skill = float(np.clip(kicking_skill, 0.92, 1.08))
    # Specialist evidence can move the empirical distance prior by only a few points.
    make_probability = float(
        np.clip(bucket.make_rate_unblocked + 0.45 * (skill - 1.0), 0.05, 0.995)
    )
    made = False if blocked else bool(rng.random() < make_probability)
    _ROWS.append(
        {
            "record_type": "field_goal",
            "distance": float(distance),
            "bucket_low": bucket.low,
            "bucket_high": bucket.high,
            "historical_attempts": bucket.attempts,
            "historical_unblocked_attempts": bucket.unblocked_attempts,
            "historical_make_rate_unblocked": bucket.make_rate_unblocked,
            "kicking_skill": skill,
            "make_probability": make_probability,
            "blocked": blocked,
            "made": made,
            "kicker_id": "" if kicker_id is None else str(kicker_id),
        }
    )
    return SpecialTeamsEvent(
        SpecialTeamsType.FIELD_GOAL,
        kick_distance=float(distance),
        blocked=blocked,
        made=made,
        kicker_id=kicker_id,
    )


def simulate_kickoff_v725(
    rng: np.random.Generator,
    *,
    returner_id: str | None = None,
    kicker_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> SpecialTeamsEvent:
    """Activate the already-compiled physical kickoff landing/return geometry."""
    event = simulate_kickoff_v633(
        rng,
        returner_id=returner_id,
        kicker_id=kicker_id,
        return_skill=return_skill,
        ecology=ecology,
        channels=_ACTIVE_CHANNEL_PRIORS,
        geometry=_ACTIVE_RETURN_GEOMETRY,
    )
    _ROWS.append(
        {
            "record_type": "kickoff",
            "distance": event.kick_distance,
            "return_yards": event.return_yards,
            "touchback": event.touchback,
            "landing": event.return_start_yardline_100,
            "muffed": event.muffed,
            "kicking_team_recovery": event.kicking_team_recovery,
            "returner_id": "" if returner_id is None else str(returner_id),
        }
    )
    return event


def resolve_turnover_return_v725(before, event, *, defense, rng, ecology=DEFAULT_CHAOS_ECOLOGY):
    """Use field-conditioned empirical fumble-return geometry; preserve V6.3 INT geometry."""
    resolved = resolve_turnover_return_v633(
        before,
        event,
        defense=defense,
        rng=rng,
        ecology=ecology,
        priors=_ACTIVE_RETURN_GEOMETRY,
    )
    _ROWS.append(
        {
            "record_type": "turnover_return",
            "turnover_kind": str(resolved.kind),
            "change_spot_yardline_100": resolved.change_spot_yardline_100,
            "return_yards": resolved.return_yards,
            "receiving_yardline_100": resolved.receiving_yardline_100,
            "touchdown": resolved.touchdown,
            "touchback": resolved.touchback,
        }
    )
    return resolved


def reset_possession_ecology_v725() -> None:
    global _ACTIVE_FIELD_GOALS, _ACTIVE_CHANNEL_PRIORS, _ACTIVE_RETURN_GEOMETRY
    _ACTIVE_FIELD_GOALS = _DEFAULT_FIELD_GOALS
    _ACTIVE_CHANNEL_PRIORS = DEFAULT_CHANNEL_PRIORS_V632
    _ACTIVE_RETURN_GEOMETRY = DEFAULT_RETURN_GEOMETRY_V633
    _ROWS.clear()


def write_possession_ecology_telemetry_v725(out: Path) -> None:
    if not _ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS, infer_schema_length=None).write_csv(
        out / "possession_ecology_v725.csv"
    )
