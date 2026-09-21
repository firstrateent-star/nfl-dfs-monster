from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from hashlib import blake2b
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality.live_state_v72 import mutation_rows_v72
from monster.sim.play_kernel import PassResult, PlayerIdentity, PlayType, TeamIdentity


@dataclass
class TeamFailurePathV722:
    team_id: str
    collapse_mode: str = "none"
    collapse_strength: float = 0.0
    finishing_friction: float = 0.0
    qb_dropbacks: int = 0
    qb_failures: int = 0
    qb_turnovers: int = 0
    qb_sacks: int = 0
    benched_qb_id: str = ""
    replacement_qb_id: str = ""
    bench_reason: str = ""


@dataclass
class FailureWorldV722:
    seed: int
    game: str
    game_drag: float
    teams: dict[str, TeamFailurePathV722] = field(default_factory=dict)


_BASE_SCRIMMAGE: Callable | None = None
_CURRENT: FailureWorldV722 | None = None
_ROWS: list[dict[str, object]] = []


def _stable_uniform(seed: int, *parts: object) -> float:
    digest = blake2b(
        ":".join(["v722-failure", str(seed), *[str(part) for part in parts]]).encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / (2**64)


def _sample_game_drag(seed: int, game: str) -> float:
    draw = _stable_uniform(seed, game, "game-drag")
    if draw >= 0.045:
        return 0.0
    severity = _stable_uniform(seed, game, "game-drag-severity")
    return float(0.10 + 0.08 * severity)


def _sample_team_state(
    seed: int,
    game: str,
    team_id: str,
    game_drag: float,
) -> TeamFailurePathV722:
    draw = _stable_uniform(seed, game, team_id, "collapse")
    severity_draw = _stable_uniform(seed, game, team_id, "collapse-severity")
    if draw < 0.055:
        mode = "hard"
        collapse = 0.16 + 0.10 * severity_draw
    elif draw < 0.14:
        mode = "drag"
        collapse = 0.06 + 0.06 * severity_draw
    else:
        mode = "none"
        collapse = 0.0

    base_finish = 0.025 + 0.045 * _stable_uniform(seed, game, team_id, "finish")
    finishing = float(
        np.clip(base_finish + 0.45 * collapse + 0.40 * game_drag, 0.02, 0.30)
    )
    return TeamFailurePathV722(
        team_id=str(team_id),
        collapse_mode=mode,
        collapse_strength=float(collapse),
        finishing_friction=finishing,
    )


def begin_failure_path_world_v722(
    *,
    seed: int,
    game: str,
    team_ids: Iterable[str] = (),
) -> FailureWorldV722:
    global _CURRENT
    game_drag = _sample_game_drag(int(seed), str(game))
    world = FailureWorldV722(seed=int(seed), game=str(game), game_drag=game_drag)
    for team_id in team_ids:
        team = _sample_team_state(world.seed, world.game, str(team_id), world.game_drag)
        world.teams[team.team_id] = team
        _ROWS.append(
            {
                "record_type": "world_state",
                "game": world.game,
                "seed": world.seed,
                "team": team.team_id,
                "player_id": "",
                "game_drag": world.game_drag,
                "collapse_mode": team.collapse_mode,
                "collapse_strength": team.collapse_strength,
                "finishing_friction": team.finishing_friction,
                "mutation": "",
                "reason": "",
            }
        )
    _CURRENT = world
    return world


def current_failure_world_v722() -> FailureWorldV722 | None:
    return _CURRENT


def team_failure_state_v722(team_id: str) -> TeamFailurePathV722 | None:
    world = _CURRENT
    if world is None:
        return None
    key = str(team_id)
    if key not in world.teams:
        world.teams[key] = _sample_team_state(world.seed, world.game, key, world.game_drag)
    return world.teams[key]


def record_pregame_availability_v722(
    team_id: str,
    *,
    active_player_ids: Iterable[str],
    inactive_player_ids: Iterable[str],
) -> None:
    world = _CURRENT
    if world is None:
        return
    active = tuple(str(x) for x in active_player_ids)
    inactive = tuple(str(x) for x in inactive_player_ids)
    team = team_failure_state_v722(team_id)
    if team is None:
        return
    _ROWS.append(
        {
            "record_type": "pregame_availability",
            "game": world.game,
            "seed": world.seed,
            "team": str(team_id),
            "player_id": "",
            "game_drag": world.game_drag,
            "collapse_mode": team.collapse_mode,
            "collapse_strength": team.collapse_strength,
            "finishing_friction": team.finishing_friction,
            "mutation": "active_set",
            "reason": f"active={len(active)};inactive={len(inactive)}",
        }
    )
    for player_id in inactive:
        _ROWS.append(
            {
                "record_type": "pregame_inactive",
                "game": world.game,
                "seed": world.seed,
                "team": str(team_id),
                "player_id": player_id,
                "game_drag": world.game_drag,
                "collapse_mode": team.collapse_mode,
                "collapse_strength": team.collapse_strength,
                "finishing_friction": team.finishing_friction,
                "mutation": "out",
                "reason": "pregame_binary_availability",
            }
        )


def _replace_qb(offense: TeamIdentity, replacement: PlayerIdentity) -> TeamIdentity:
    starter_id = offense.quarterback.player_id
    rushers: list[PlayerIdentity] = []
    replaced_rusher = False
    for rusher in offense.rushers:
        if rusher.player_id == starter_id:
            rushers.append(replace(replacement, usage_weight=rusher.usage_weight))
            replaced_rusher = True
        else:
            rushers.append(rusher)
    if not replaced_rusher and all(
        rusher.player_id != replacement.player_id for rusher in rushers
    ):
        rushers.append(replace(replacement, usage_weight=0.001))
    return replace(offense, quarterback=replacement, rushers=tuple(rushers))


def _apply_failure_identity_v722(
    offense: TeamIdentity,
    state: object,
) -> TeamIdentity:
    world = _CURRENT
    if world is None:
        return offense
    team = team_failure_state_v722(offense.team_id)
    if team is None:
        return offense

    adjusted = offense
    if team.replacement_qb_id:
        from monster.reality.game_script_v721 import _QB_DEPTH

        replacement = next(
            (
                player
                for player in _QB_DEPTH.get(str(offense.team_id), ())
                if player.player_id == team.replacement_qb_id
            ),
            None,
        )
        if (
            replacement is not None
            and offense.quarterback.player_id != replacement.player_id
        ):
            adjusted = _replace_qb(adjusted, replacement)

    structural = float(
        np.clip(world.game_drag + team.collapse_strength, 0.0, 0.36)
    )
    field_position = float(getattr(state, "yardline_100", 50.0))
    finish_zone = float(np.clip((field_position - 55.0) / 35.0, 0.0, 1.0))
    finish = float(
        np.clip(team.finishing_friction * finish_zone, 0.0, 0.30)
    )

    quarterback = replace(
        adjusted.quarterback,
        efficiency=float(
            max(
                adjusted.quarterback.efficiency
                * (1.0 - 0.30 * structural - 0.18 * finish),
                0.55,
            )
        ),
        turnover_security=float(
            max(
                adjusted.quarterback.turnover_security
                * (1.0 - 0.45 * structural - 0.35 * finish),
                0.50,
            )
        ),
    )
    return replace(
        adjusted,
        quarterback=quarterback,
        pass_efficiency=float(
            max(
                adjusted.pass_efficiency
                * (1.0 - 0.60 * structural - 0.60 * finish),
                0.52,
            )
        ),
        rush_efficiency=float(
            max(
                adjusted.rush_efficiency
                * (1.0 - 0.45 * structural - 0.48 * finish),
                0.58,
            )
        ),
        pass_protection=float(
            max(
                adjusted.pass_protection
                * (1.0 - 0.55 * structural - 0.35 * finish),
                0.55,
            )
        ),
        run_blocking=float(
            max(
                adjusted.run_blocking
                * (1.0 - 0.30 * structural - 0.25 * finish),
                0.62,
            )
        ),
        field_goal_skill=float(
            max(adjusted.field_goal_skill * (1.0 - 0.25 * finish), 0.70)
        ),
    )


def _record_live_mutations(start_index: int) -> None:
    world = _CURRENT
    if world is None:
        return
    rows = mutation_rows_v72()
    for row in rows[start_index:]:
        team_id = str(row.get("team") or "")
        team = team_failure_state_v722(team_id) if team_id else None
        _ROWS.append(
            {
                "record_type": "live_availability",
                "game": world.game,
                "seed": world.seed,
                "team": team_id,
                "player_id": str(row.get("player_id") or ""),
                "game_drag": world.game_drag,
                "collapse_mode": "" if team is None else team.collapse_mode,
                "collapse_strength": 0.0 if team is None else team.collapse_strength,
                "finishing_friction": 0.0 if team is None else team.finishing_friction,
                "mutation": str(row.get("mutation") or ""),
                "reason": "in_game_live_state",
            }
        )


def _update_qb_performance_v722(
    *,
    state: object,
    offense: TeamIdentity,
    event: object,
) -> None:
    world = _CURRENT
    if world is None:
        return
    team = team_failure_state_v722(offense.team_id)
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
    if quarter < 3 or margin > -14 or team.qb_dropbacks < 14:
        return

    failure_rate = team.qb_failures / max(team.qb_dropbacks, 1)
    severe = (
        team.qb_turnovers >= 2
        or (team.qb_sacks >= 3 and failure_rate >= 0.58)
        or (team.qb_dropbacks >= 20 and failure_rate >= 0.62)
    )
    if not severe:
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
        f"q{quarter};margin={margin};dropbacks={team.qb_dropbacks};"
        f"failure_rate={failure_rate:.3f};turnovers={team.qb_turnovers};"
        f"sacks={team.qb_sacks}"
    )
    _ROWS.append(
        {
            "record_type": "qb_bench",
            "game": world.game,
            "seed": world.seed,
            "team": str(offense.team_id),
            "player_id": offense.quarterback.player_id,
            "game_drag": world.game_drag,
            "collapse_mode": team.collapse_mode,
            "collapse_strength": team.collapse_strength,
            "finishing_friction": team.finishing_friction,
            "mutation": f"bench->{replacement.player_id}",
            "reason": team.bench_reason,
        }
    )


def configure_failure_path_scrimmage_v722(fn: Callable) -> None:
    global _BASE_SCRIMMAGE
    if fn is not simulate_scrimmage_play_v722:
        _BASE_SCRIMMAGE = fn


def simulate_scrimmage_play_v722(
    state,
    offense,
    defense_strength,
    rng,
    defense=None,
):
    if _BASE_SCRIMMAGE is None:
        raise RuntimeError(
            "V7.2.2 failure-path base scrimmage runtime is not configured"
        )

    active = _apply_failure_identity_v722(offense, state)
    mutation_start = len(mutation_rows_v72())
    event = _BASE_SCRIMMAGE(
        state,
        active,
        defense_strength,
        rng,
        defense=defense,
    )
    _record_live_mutations(mutation_start)
    _update_qb_performance_v722(
        state=state,
        offense=active,
        event=event,
    )
    return event


def failure_path_rows_v722() -> list[dict[str, object]]:
    return list(_ROWS)


def reset_failure_paths_v722() -> None:
    global _CURRENT
    _CURRENT = None
    _ROWS.clear()


def write_failure_path_telemetry_v722(out: Path) -> None:
    if not _ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_csv(out / "failure_paths_v722.csv")
