from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality import failure_paths_v722 as v722
from monster.reality.world_availability_v722 import materialize_world_pools_v722
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType, TeamIdentity


@dataclass(frozen=True)
class CalibrationV723:
    game_drag_probability: float = 0.080
    hard_collapse_probability: float = 0.075
    any_collapse_probability: float = 0.240
    hard_collapse_min: float = 0.20
    hard_collapse_span: float = 0.12
    drag_collapse_min: float = 0.08
    drag_collapse_span: float = 0.08


CALIBRATION = CalibrationV723()
_BASE_SCRIMMAGE: Callable | None = None
_ROWS: list[dict[str, object]] = []


def _stable_uniform(seed: int, *parts: object) -> float:
    digest = blake2b(
        ":".join(["v723-calibration", str(seed), *[str(part) for part in parts]]).encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / (2**64)


def recalibrate_current_failure_world_v723(
    *,
    seed: int,
    game: str,
    team_ids: Iterable[str],
) -> None:
    world = v722.current_failure_world_v722()
    if world is None:
        raise RuntimeError("V7.2.3 requires an initialized V7.2.2 failure world")

    game_drag_draw = _stable_uniform(seed, game, "game-drag")
    if game_drag_draw < CALIBRATION.game_drag_probability:
        severity = _stable_uniform(seed, game, "game-drag-severity")
        world.game_drag = float(0.08 + 0.08 * severity)
    else:
        world.game_drag = 0.0

    for team_id in team_ids:
        team = v722.team_failure_state_v722(str(team_id))
        if team is None:
            continue
        draw = _stable_uniform(seed, game, team_id, "collapse")
        severity = _stable_uniform(seed, game, team_id, "collapse-severity")
        if draw < CALIBRATION.hard_collapse_probability:
            team.collapse_mode = "hard"
            team.collapse_strength = float(
                CALIBRATION.hard_collapse_min
                + CALIBRATION.hard_collapse_span * severity
            )
        elif draw < CALIBRATION.any_collapse_probability:
            team.collapse_mode = "drag"
            team.collapse_strength = float(
                CALIBRATION.drag_collapse_min
                + CALIBRATION.drag_collapse_span * severity
            )
        else:
            team.collapse_mode = "none"
            team.collapse_strength = 0.0

        base_finish = 0.030 + 0.040 * _stable_uniform(
            seed, game, team_id, "finish"
        )
        team.finishing_friction = float(
            np.clip(
                base_finish
                + 0.58 * team.collapse_strength
                + 0.38 * world.game_drag,
                0.03,
                0.38,
            )
        )
        _ROWS.append(
            {
                "record_type": "world_calibration",
                "game": str(game),
                "seed": int(seed),
                "team": str(team_id),
                "quarter": 0,
                "seconds_remaining": 0,
                "down": 0,
                "yardline_100": 0.0,
                "collapse_mode": team.collapse_mode,
                "collapse_strength": team.collapse_strength,
                "game_drag": world.game_drag,
                "finishing_friction": team.finishing_friction,
                "stall_probability": 0.0,
                "mutation": "",
                "reason": "",
            }
        )


def materialize_world_pools_v723(
    pools,
    *,
    seed: int,
    game: str,
):
    result = materialize_world_pools_v722(pools, seed=seed, game=game)
    recalibrate_current_failure_world_v723(
        seed=seed,
        game=game,
        team_ids=tuple(sorted(pools)),
    )
    return result


def _stall_probability(
    *,
    state: object,
    team: v722.TeamFailurePathV722,
    event: PlayEvent,
) -> float:
    if event.play_type not in {PlayType.RUN, PlayType.PASS}:
        return 0.0
    if event.turnover or event.fumbler_id is not None:
        return 0.0
    yardline = float(getattr(state, "yardline_100", 0.0))
    if yardline < 65.0:
        return 0.0

    down = int(getattr(state, "down", 1))
    distance = float(getattr(state, "distance", 10.0))
    conversion = event.yards >= distance
    high_leverage = down >= 3 and conversion
    if not event.touchdown and not high_leverage:
        return 0.0

    territory = float(np.clip((yardline - 65.0) / 35.0, 0.0, 1.0))
    leverage = 0.22 + 0.48 * territory
    if down >= 3:
        leverage += 0.28
    if event.touchdown:
        leverage += 0.18
    return float(
        np.clip(team.finishing_friction * leverage, 0.0, 0.28)
    )


def _failed_run_event(state: object, event: PlayEvent) -> PlayEvent:
    distance = float(getattr(state, "distance", 10.0))
    down = int(getattr(state, "down", 1))
    yardline = float(getattr(state, "yardline_100", 0.0))
    yards_to_goal = max(100.0 - yardline, 0.0)

    if down >= 3:
        cap = max(distance - 1.0, 0.0)
    else:
        cap = max(yards_to_goal - 1.0, 0.0)
    yards = float(min(max(event.yards, -2.0), cap))
    return replace(
        event,
        yards=yards,
        touchdown=False,
        turnover=False,
        fumbler_id=None,
        stuffed=True,
        yards_before_contact=min(float(event.yards_before_contact), max(yards, 0.0)),
        yards_after_contact=0.0,
    )


def _failed_scramble_event(state: object, event: PlayEvent) -> PlayEvent:
    distance = float(getattr(state, "distance", 10.0))
    yardline = float(getattr(state, "yardline_100", 0.0))
    yards_to_goal = max(100.0 - yardline, 0.0)
    cap = min(
        max(distance - 1.0, 0.0),
        max(yards_to_goal - 1.0, 0.0),
    )
    yards = float(min(max(event.yards, -2.0), cap))
    return replace(
        event,
        yards=yards,
        touchdown=False,
        turnover=False,
        fumbler_id=None,
        pass_result=PassResult.SCRAMBLE,
        target_id=None,
        air_yards=0.0,
        yards_after_catch=0.0,
        yards_before_contact=min(float(event.yards_before_contact), max(yards, 0.0)),
        yards_after_contact=0.0,
    )


def _failed_pass_event(
    *,
    seed: int,
    state: object,
    offense: TeamIdentity,
    event: PlayEvent,
) -> tuple[PlayEvent, str]:
    world = v722.current_failure_world_v722()
    game = "" if world is None else world.game
    quarter = int(getattr(state, "quarter", 1))
    seconds = int(getattr(state, "seconds_remaining", 0))
    draw = _stable_uniform(
        seed,
        game,
        offense.team_id,
        quarter,
        seconds,
        int(getattr(state, "down", 1)),
        "stall-kind",
    )
    if draw < 0.34:
        loss = 4.0 + 5.0 * _stable_uniform(
            seed, game, offense.team_id, quarter, seconds, "stall-sack-yards"
        )
        return (
            replace(
                event,
                yards=-float(loss),
                target_id=None,
                pass_result=PassResult.SACK,
                touchdown=False,
                turnover=False,
                fumbler_id=None,
                pressured=True,
                air_yards=0.0,
                yards_after_catch=0.0,
            ),
            "sack",
        )
    return (
        replace(
            event,
            yards=0.0,
            pass_result=PassResult.INCOMPLETE,
            touchdown=False,
            turnover=False,
            fumbler_id=None,
            air_yards=0.0,
            yards_after_catch=0.0,
        ),
        "incomplete",
    )


def apply_drive_finishing_v723(
    *,
    state: object,
    offense: TeamIdentity,
    event: PlayEvent,
) -> PlayEvent:
    world = v722.current_failure_world_v722()
    if world is None:
        return event
    team = v722.team_failure_state_v722(offense.team_id)
    if team is None:
        return event

    probability = _stall_probability(state=state, team=team, event=event)
    if probability <= 0.0:
        return event

    seed = int(world.seed)
    draw = _stable_uniform(
        seed,
        world.game,
        offense.team_id,
        int(getattr(state, "quarter", 1)),
        int(getattr(state, "seconds_remaining", 0)),
        int(getattr(state, "down", 1)),
        round(float(getattr(state, "yardline_100", 0.0)), 1),
        event.play_type.value,
        event.passer_id or "",
        event.rusher_id or "",
        event.target_id or "",
        "finish-stall",
    )
    if draw >= probability:
        return event

    if event.play_type == PlayType.PASS and event.pass_result == PassResult.SCRAMBLE:
        adjusted = _failed_scramble_event(state, event)
        kind = "scramble_short"
    elif event.play_type == PlayType.PASS:
        adjusted, kind = _failed_pass_event(
            seed=seed,
            state=state,
            offense=offense,
            event=event,
        )
    else:
        adjusted = _failed_run_event(state, event)
        kind = "stuff"

    _ROWS.append(
        {
            "record_type": "finish_stall",
            "game": world.game,
            "seed": world.seed,
            "team": offense.team_id,
            "quarter": int(getattr(state, "quarter", 1)),
            "seconds_remaining": int(getattr(state, "seconds_remaining", 0)),
            "down": int(getattr(state, "down", 1)),
            "yardline_100": float(getattr(state, "yardline_100", 0.0)),
            "collapse_mode": team.collapse_mode,
            "collapse_strength": team.collapse_strength,
            "game_drag": world.game_drag,
            "finishing_friction": team.finishing_friction,
            "stall_probability": probability,
            "mutation": kind,
            "reason": (
                "touchdown_nullified"
                if event.touchdown
                else "high_leverage_conversion_failed"
            ),
        }
    )
    return adjusted


def update_qb_performance_v723(
    *,
    state: object,
    offense: TeamIdentity,
    event: object,
) -> None:
    world = v722.current_failure_world_v722()
    if world is None:
        return
    team = v722.team_failure_state_v722(offense.team_id)
    if team is None or team.replacement_qb_id:
        return
    if getattr(event, "play_type", None) != PlayType.PASS:
        return
    passer_id = str(getattr(event, "passer_id", "") or "")
    if passer_id != offense.quarterback.player_id:
        return

    team.qb_dropbacks += 1
    result = getattr(event, "pass_result", None)
    if result in {
        PassResult.INCOMPLETE,
        PassResult.SACK,
        PassResult.INTERCEPTION,
    }:
        team.qb_failures += 1
    if result == PassResult.SACK:
        team.qb_sacks += 1
    if bool(getattr(event, "turnover", False)) or result == PassResult.INTERCEPTION:
        team.qb_turnovers += 1

    quarter = int(getattr(state, "quarter", 1))
    margin = int(getattr(state, "score_margin_for_offense", 0))
    failure_rate = team.qb_failures / max(team.qb_dropbacks, 1)

    catastrophic_q3 = (
        quarter == 3
        and margin <= -28
        and team.qb_dropbacks >= 24
        and team.qb_turnovers >= 3
        and failure_rate >= 0.62
    )
    catastrophic_q4 = (
        quarter >= 4
        and margin <= -17
        and team.qb_dropbacks >= 24
        and (
            team.qb_turnovers >= 3
            or (
                team.qb_dropbacks >= 30
                and team.qb_sacks >= 4
                and failure_rate >= 0.70
            )
        )
    )
    if not (catastrophic_q3 or catastrophic_q4):
        return

    willingness = _stable_uniform(
        world.seed,
        world.game,
        offense.team_id,
        offense.quarterback.player_id,
        quarter,
        int(getattr(state, "seconds_remaining", 0)),
        "bench-willingness",
    )
    if willingness >= 0.62:
        return

    from monster.reality.game_script_v721 import _backup_qb

    replacement = _backup_qb(
        offense.team_id,
        offense.quarterback.player_id,
    )
    if replacement is None:
        return

    team.benched_qb_id = offense.quarterback.player_id
    team.replacement_qb_id = replacement.player_id
    team.bench_reason = (
        f"v723;q{quarter};margin={margin};dropbacks={team.qb_dropbacks};"
        f"failure_rate={failure_rate:.3f};turnovers={team.qb_turnovers};"
        f"sacks={team.qb_sacks};willingness={willingness:.3f}"
    )
    _ROWS.append(
        {
            "record_type": "qb_bench",
            "game": world.game,
            "seed": world.seed,
            "team": offense.team_id,
            "quarter": quarter,
            "seconds_remaining": int(getattr(state, "seconds_remaining", 0)),
            "down": int(getattr(state, "down", 1)),
            "yardline_100": float(getattr(state, "yardline_100", 0.0)),
            "collapse_mode": team.collapse_mode,
            "collapse_strength": team.collapse_strength,
            "game_drag": world.game_drag,
            "finishing_friction": team.finishing_friction,
            "stall_probability": 0.0,
            "mutation": f"bench->{replacement.player_id}",
            "reason": team.bench_reason,
        }
    )


def configure_scrimmage_v723(fn: Callable) -> None:
    global _BASE_SCRIMMAGE
    if fn is not simulate_scrimmage_play_v723:
        _BASE_SCRIMMAGE = fn


def simulate_scrimmage_play_v723(
    state,
    offense,
    defense_strength,
    rng,
    defense=None,
):
    if _BASE_SCRIMMAGE is None:
        raise RuntimeError("V7.2.3 base scrimmage runtime is not configured")
    event = _BASE_SCRIMMAGE(
        state,
        offense,
        defense_strength,
        rng,
        defense=defense,
    )
    return apply_drive_finishing_v723(
        state=state,
        offense=offense,
        event=event,
    )


def reset_v723_state() -> None:
    _ROWS.clear()


def calibration_rows_v723() -> list[dict[str, object]]:
    return list(_ROWS)


def write_calibration_telemetry_v723(out: Path) -> None:
    if not _ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_csv(out / "failure_calibration_v723.csv")
