from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality import failure_paths_v722 as v722
from monster.reality import failure_paths_v723 as v723
from monster.sim.play_kernel import PlayEvent, PlayType


@dataclass(frozen=True)
class ConsequenceCalibrationV724:
    normal_finish_min: float = 0.012
    normal_finish_span: float = 0.012
    drag_finish_min: float = 0.190
    drag_finish_span: float = 0.100
    hard_finish_min: float = 0.420
    hard_finish_span: float = 0.180
    normal_game_drag_weight: float = 0.20
    drag_game_drag_weight: float = 0.32
    hard_game_drag_weight: float = 0.40
    max_finishing_friction: float = 0.64
    max_stall_probability: float = 0.68


CALIBRATION = ConsequenceCalibrationV724()
_ROWS: list[dict[str, object]] = []


def _stable_uniform(seed: int, *parts: object) -> float:
    digest = blake2b(
        ":".join(["v724-consequence", str(seed), *[str(part) for part in parts]]).encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / (2**64)


def _mode_severity(mode: str, collapse_strength: float) -> float:
    if mode == "hard":
        return float(np.clip((collapse_strength - 0.20) / 0.12, 0.0, 1.0))
    if mode == "drag":
        return float(np.clip((collapse_strength - 0.08) / 0.08, 0.0, 1.0))
    return 0.0


def reweight_finishing_consequence_v724(*, seed: int, game: str) -> None:
    world = v722.current_failure_world_v722()
    if world is None:
        raise RuntimeError("V7.2.4 requires an initialized V7.2.3 failure world")

    for team_id, team in sorted(world.teams.items()):
        previous = float(team.finishing_friction)
        severity = _mode_severity(team.collapse_mode, team.collapse_strength)
        texture = _stable_uniform(seed, game, team_id, "finish-texture")

        if team.collapse_mode == "hard":
            base = (
                CALIBRATION.hard_finish_min
                + CALIBRATION.hard_finish_span * (0.75 * severity + 0.25 * texture)
            )
            game_drag = CALIBRATION.hard_game_drag_weight * world.game_drag
        elif team.collapse_mode == "drag":
            base = (
                CALIBRATION.drag_finish_min
                + CALIBRATION.drag_finish_span * (0.75 * severity + 0.25 * texture)
            )
            game_drag = CALIBRATION.drag_game_drag_weight * world.game_drag
        else:
            base = CALIBRATION.normal_finish_min + CALIBRATION.normal_finish_span * texture
            game_drag = CALIBRATION.normal_game_drag_weight * world.game_drag

        team.finishing_friction = float(
            np.clip(
                base + game_drag,
                0.0,
                CALIBRATION.max_finishing_friction,
            )
        )
        _ROWS.append(
            {
                "record_type": "world_consequence",
                "game": str(game),
                "seed": int(seed),
                "team": str(team_id),
                "collapse_mode": team.collapse_mode,
                "collapse_strength": float(team.collapse_strength),
                "game_drag": float(world.game_drag),
                "v723_finishing_friction": previous,
                "v724_finishing_friction": float(team.finishing_friction),
                "mode_severity": severity,
            }
        )


def materialize_world_pools_v724(pools, *, seed: int, game: str):
    result = v723.materialize_world_pools_v723(pools, seed=seed, game=game)
    reweight_finishing_consequence_v724(seed=seed, game=game)
    return result


def _stall_probability_v724(
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

    world = v722.current_failure_world_v722()
    game_drag = 0.0 if world is None else float(world.game_drag)
    if team.collapse_mode == "hard":
        mode_multiplier = 1.10
    elif team.collapse_mode == "drag":
        mode_multiplier = 1.00
    elif game_drag > 0.0:
        mode_multiplier = 0.75
    else:
        mode_multiplier = 0.35

    return float(
        np.clip(
            team.finishing_friction * leverage * mode_multiplier,
            0.0,
            CALIBRATION.max_stall_probability,
        )
    )


def install_drive_consequence_v724() -> None:
    v723._stall_probability = _stall_probability_v724


def reset_v724_state() -> None:
    _ROWS.clear()


def consequence_rows_v724() -> list[dict[str, object]]:
    return list(_ROWS)


def write_consequence_telemetry_v724(out: Path) -> None:
    if not _ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_csv(out / "failure_consequence_v724.csv")
