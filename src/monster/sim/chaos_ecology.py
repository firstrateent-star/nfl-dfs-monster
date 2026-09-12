from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

from monster.sim.football_state import FootballState, mirror_field
from monster.sim.play_kernel import PassResult, PlayEvent

if TYPE_CHECKING:
    from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit


class ReturnKind(StrEnum):
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    PUNT = "punt"
    KICKOFF = "kickoff"
    BLOCKED_PUNT = "blocked_punt"
    BLOCKED_FIELD_GOAL = "blocked_field_goal"


@dataclass(frozen=True)
class ChaosEcology:
    """League-level priors for rare, high-leverage change-of-possession events.

    These values describe football geometry, not score targets. The event engine still has to
    create the turnover/kick, determine where the ball changes hands, and earn any return TD by
    physically traversing the remaining field. Historical play-by-play can replace the defaults
    through ``from_policy_row`` without changing the runtime contract.
    """

    interception_zero_return_rate: float = 0.22
    interception_return_mean: float = 11.5
    interception_return_sd: float = 12.0
    interception_40_plus_rate: float = 0.045
    fumble_zero_return_rate: float = 0.50
    fumble_return_mean: float = 7.0
    fumble_return_sd: float = 9.0
    fumble_40_plus_rate: float = 0.020
    punt_zero_return_rate: float = 0.46
    punt_return_mean: float = 8.8
    punt_return_sd: float = 10.0
    punt_40_plus_rate: float = 0.018
    kickoff_return_mean: float = 24.5
    kickoff_return_sd: float = 10.5
    kickoff_40_plus_rate: float = 0.028
    punt_muff_rate: float = 0.012
    punt_muff_kicking_recovery_rate: float = 0.48
    kickoff_muff_rate: float = 0.004
    kickoff_muff_kicking_recovery_rate: float = 0.45
    blocked_punt_rate: float = 0.012
    blocked_field_goal_rate: float = 0.010
    kickoff_touchback_rate: float = 0.60
    kickoff_touchback_yardline: float = 35.0


DEFAULT_CHAOS_ECOLOGY = ChaosEcology()


def _number(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(out) else out


def from_policy_row(row: Mapping[str, object] | None) -> ChaosEcology:
    """Build runtime priors from a compiled historical row, preserving safe defaults."""
    if not row:
        return DEFAULT_CHAOS_ECOLOGY
    d = DEFAULT_CHAOS_ECOLOGY
    return ChaosEcology(
        interception_zero_return_rate=float(np.clip(_number(row, "interception_zero_return_rate", d.interception_zero_return_rate), 0.0, 0.90)),
        interception_return_mean=float(np.clip(_number(row, "interception_return_mean", d.interception_return_mean), 1.0, 40.0)),
        interception_return_sd=float(np.clip(_number(row, "interception_return_sd", d.interception_return_sd), 1.0, 40.0)),
        interception_40_plus_rate=float(np.clip(_number(row, "interception_40_plus_rate", d.interception_40_plus_rate), 0.0, 0.25)),
        fumble_zero_return_rate=float(np.clip(_number(row, "fumble_zero_return_rate", d.fumble_zero_return_rate), 0.0, 0.95)),
        fumble_return_mean=float(np.clip(_number(row, "fumble_return_mean", d.fumble_return_mean), 1.0, 35.0)),
        fumble_return_sd=float(np.clip(_number(row, "fumble_return_sd", d.fumble_return_sd), 1.0, 35.0)),
        fumble_40_plus_rate=float(np.clip(_number(row, "fumble_40_plus_rate", d.fumble_40_plus_rate), 0.0, 0.20)),
        punt_zero_return_rate=float(np.clip(_number(row, "punt_zero_return_rate", d.punt_zero_return_rate), 0.0, 0.95)),
        punt_return_mean=float(np.clip(_number(row, "punt_return_mean", d.punt_return_mean), 1.0, 30.0)),
        punt_return_sd=float(np.clip(_number(row, "punt_return_sd", d.punt_return_sd), 1.0, 35.0)),
        punt_40_plus_rate=float(np.clip(_number(row, "punt_40_plus_rate", d.punt_40_plus_rate), 0.0, 0.20)),
        kickoff_return_mean=float(np.clip(_number(row, "kickoff_return_mean", d.kickoff_return_mean), 8.0, 45.0)),
        kickoff_return_sd=float(np.clip(_number(row, "kickoff_return_sd", d.kickoff_return_sd), 2.0, 35.0)),
        kickoff_40_plus_rate=float(np.clip(_number(row, "kickoff_40_plus_rate", d.kickoff_40_plus_rate), 0.0, 0.20)),
        punt_muff_rate=float(np.clip(_number(row, "punt_muff_rate", d.punt_muff_rate), 0.0, 0.08)),
        punt_muff_kicking_recovery_rate=float(np.clip(_number(row, "punt_muff_kicking_recovery_rate", d.punt_muff_kicking_recovery_rate), 0.0, 1.0)),
        kickoff_muff_rate=float(np.clip(_number(row, "kickoff_muff_rate", d.kickoff_muff_rate), 0.0, 0.05)),
        kickoff_muff_kicking_recovery_rate=float(np.clip(_number(row, "kickoff_muff_kicking_recovery_rate", d.kickoff_muff_kicking_recovery_rate), 0.0, 1.0)),
        blocked_punt_rate=float(np.clip(_number(row, "blocked_punt_rate", d.blocked_punt_rate), 0.0, 0.08)),
        blocked_field_goal_rate=float(np.clip(_number(row, "blocked_field_goal_rate", d.blocked_field_goal_rate), 0.0, 0.08)),
        kickoff_touchback_rate=float(np.clip(_number(row, "kickoff_touchback_rate", d.kickoff_touchback_rate), 0.05, 0.95)),
        kickoff_touchback_yardline=float(np.clip(_number(row, "kickoff_touchback_yardline", d.kickoff_touchback_yardline), 20.0, 40.0)),
    )


@dataclass(frozen=True)
class ReturnEvent:
    kind: ReturnKind
    original_offense_team_id: str
    return_team_id: str
    returner_id: str | None
    change_spot_yardline_100: float
    return_yards: float
    receiving_yardline_100: float
    touchdown: bool
    touchback: bool = False
    muffed: bool = False
    kicking_team_recovery: bool = False


def sample_return_yards(
    *,
    mean: float,
    sd: float,
    zero_rate: float,
    forty_plus_rate: float,
    return_skill: float,
    rng: np.random.Generator,
    maximum: float = 100.0,
) -> float:
    """Sample ordinary and breakaway return branches without directly sampling a TD.

    A return touchdown occurs only when the sampled return covers the geometric distance to
    the goal line. The 40+ branch preserves the rare heavy tail that a single normal draw
    tends to erase.
    """
    if rng.random() < float(np.clip(zero_rate, 0.0, 0.98)):
        return 0.0
    skill = float(np.clip(return_skill, 0.72, 1.32))
    if rng.random() < float(np.clip(forty_plus_rate, 0.0, 0.30)):
        yards = 40.0 + rng.exponential(max(10.0 * skill, 2.0))
        return float(np.clip(yards, 40.0, maximum))

    live_mean = max(mean * skill, 0.5)
    live_sd = max(sd * sqrt(skill), 0.75)
    shape = max((live_mean / live_sd) ** 2, 0.25)
    scale = max((live_sd**2) / live_mean, 0.05)
    return float(np.clip(rng.gamma(shape, scale), 0.0, maximum))


def _returner(defense: DefensiveUnit | None, player_id: str | None) -> DefensiveIdentity | None:
    if defense is None:
        return None
    defenders = defense.coverage + defense.front
    if player_id is not None:
        found = next((item for item in defenders if item.player_id == player_id), None)
        if found is not None:
            return found
    return max(
        defenders,
        key=lambda item: max(item.snap_weight, 0.01) * max(item.returning, 0.50),
        default=None,
    )


def resolve_turnover_return(
    before: FootballState,
    event: PlayEvent,
    *,
    defense: DefensiveUnit | None,
    rng: np.random.Generator,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> ReturnEvent:
    """Resolve where a turnover happens and what the defense does with the live ball."""
    if not event.turnover:
        raise ValueError("turnover return resolution requires a turnover event")

    interception = event.pass_result == PassResult.INTERCEPTION
    kind = ReturnKind.INTERCEPTION if interception else ReturnKind.FUMBLE
    if interception:
        raw_spot = before.yardline_100 + float(event.air_yards)
        # A defender who secures a pass in the end zone can simply take the touchback.
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
        forty = ecology.interception_40_plus_rate
    else:
        spot = float(np.clip(before.yardline_100 + float(event.yards), 1.0, 99.0))
        zero_rate = ecology.fumble_zero_return_rate
        mean = ecology.fumble_return_mean
        sd = ecology.fumble_return_sd
        forty = ecology.fumble_40_plus_rate

    defender = _returner(defense, event.primary_defender_id)
    returner_id = None if defender is None else defender.player_id
    return_skill = 1.0 if defender is None else float(defender.returning)
    start = mirror_field(spot)
    distance_to_goal = 100.0 - start
    yards = sample_return_yards(
        mean=mean,
        sd=sd,
        zero_rate=zero_rate,
        forty_plus_rate=forty,
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
