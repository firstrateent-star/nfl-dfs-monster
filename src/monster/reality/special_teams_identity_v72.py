from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import tanh
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality.participation_authority_v72 import (
    participation_multiplier,
    role_for,
)


@dataclass(frozen=True)
class SpecialTeamsParticipantV72:
    player_id: str
    team_id: str
    position: str
    depth_rank: int
    special_share: float
    active_probability: float
    madden_return: float | None
    madden_kick_power: float | None
    madden_kick_accuracy: float | None


_PLAYERS: dict[str, tuple[SpecialTeamsParticipantV72, ...]] = {}
_CURRENT_OFFENSE: str | None = None
_CURRENT_DEFENSE: str | None = None
_BASE_PUNT: Callable | None = None
_BASE_FIELD_GOAL: Callable | None = None
_BASE_KICKOFF: Callable | None = None
_BASE_KICKOFF_LOOP: Callable | None = None
_EVENT_ROWS: list[dict[str, object]] = []


def configure_special_teams_identities_v72(personnel: pl.DataFrame) -> None:
    by_team: dict[str, list[SpecialTeamsParticipantV72]] = {}
    for row in personnel.to_dicts():
        player_id = str(
            row.get("gsis_id")
            or row.get("pfr_id")
            or row.get("display_name")
            or ""
        )
        team_id = str(row.get("team_id") or "")
        if not player_id or not team_id:
            continue
        try:
            depth_rank = int(float(row.get("depth_rank") or 0))
        except (TypeError, ValueError):
            depth_rank = 0

        def opt(name: str, source: dict = row) -> float | None:
            value = source.get(name)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        by_team.setdefault(team_id, []).append(
            SpecialTeamsParticipantV72(
                player_id=player_id,
                team_id=team_id,
                position=str(row.get("position") or "").upper(),
                depth_rank=depth_rank,
                special_share=float(
                    np.clip(
                        float(row.get("conditional_special_teams_snap_share") or 0.0),
                        0.0,
                        1.0,
                    )
                ),
                active_probability=float(
                    np.clip(
                        float(row.get("game_day_active_probability") or 0.0),
                        0.0,
                        1.0,
                    )
                ),
                madden_return=opt("madden_return"),
                madden_kick_power=opt("madden_kick_power"),
                madden_kick_accuracy=opt("madden_kick_accuracy"),
            )
        )
    global _PLAYERS
    _PLAYERS = {
        team: tuple(players)
        for team, players in by_team.items()
    }


def configure_base_special_teams_hooks_v72(
    *,
    punt: Callable,
    field_goal: Callable,
    kickoff: Callable,
    kickoff_loop: Callable,
) -> None:
    global _BASE_PUNT, _BASE_FIELD_GOAL, _BASE_KICKOFF, _BASE_KICKOFF_LOOP
    _BASE_PUNT = punt
    _BASE_FIELD_GOAL = field_goal
    _BASE_KICKOFF = kickoff
    _BASE_KICKOFF_LOOP = kickoff_loop


def set_play_context_v72(offense_team: str, defense_team: str) -> None:
    global _CURRENT_OFFENSE, _CURRENT_DEFENSE
    _CURRENT_OFFENSE = str(offense_team)
    _CURRENT_DEFENSE = str(defense_team)


def set_kick_context_v72(kicking_team: str, receiving_team: str) -> None:
    set_play_context_v72(kicking_team, receiving_team)


def _rank(participant: SpecialTeamsParticipantV72) -> tuple[float, float, str]:
    depth = 0.0
    if participant.depth_rank == 1:
        depth = 1.0
    elif participant.depth_rank == 2:
        depth = 0.62
    elif participant.depth_rank == 3:
        depth = 0.36
    current = participation_multiplier(participant.player_id, unit="special")
    return (
        -(0.55 * current + 0.30 * participant.special_share + 0.15 * depth),
        -participant.active_probability,
        participant.player_id,
    )


def _specialist(team_id: str | None, positions: set[str]) -> SpecialTeamsParticipantV72 | None:
    if not team_id:
        return None
    candidates = [
        player
        for player in _PLAYERS.get(str(team_id), ())
        if player.position in positions
        and player.active_probability >= 0.10
    ]
    if not candidates:
        return None
    return min(candidates, key=_rank)


def _returner(team_id: str | None) -> SpecialTeamsParticipantV72 | None:
    if not team_id:
        return None
    candidates = [
        player
        for player in _PLAYERS.get(str(team_id), ())
        if player.position in {"WR", "RB", "CB", "DB", "S", "FS", "SS"}
        and player.active_probability >= 0.10
        and player.special_share > 0.01
    ]
    if not candidates:
        return None

    def score(player: SpecialTeamsParticipantV72) -> tuple[float, str]:
        return_rating = 78.0 if player.madden_return is None else player.madden_return
        value = (
            0.54 * player.special_share
            + 0.28 * participation_multiplier(player.player_id, unit="special")
            + 0.18 * np.clip((return_rating - 65.0) / 35.0, 0.0, 1.0)
        )
        return (-float(value), player.player_id)

    return min(candidates, key=score)


def _kick_skill(player: SpecialTeamsParticipantV72 | None) -> float:
    if player is None:
        return 1.0
    values = [
        value
        for value in (player.madden_kick_accuracy, player.madden_kick_power)
        if value is not None
    ]
    if not values:
        return 1.0
    rating = float(np.mean(values))
    return float(np.clip(1.0 + 0.08 * tanh((rating - 78.0) / 10.0), 0.92, 1.08))


def _return_skill(player: SpecialTeamsParticipantV72 | None) -> float:
    if player is None or player.madden_return is None:
        return 1.0
    return float(
        np.clip(
            1.0 + 0.14 * tanh((player.madden_return - 78.0) / 11.0),
            0.86,
            1.14,
        )
    )


def simulate_punt_v72(rng, **kwargs):
    if _BASE_PUNT is None:
        raise RuntimeError("v7.2 base punt hook is not configured")
    punter = _specialist(_CURRENT_OFFENSE, {"P"})
    returner = _returner(_CURRENT_DEFENSE)
    kwargs["punter_id"] = None if punter is None else punter.player_id
    kwargs["returner_id"] = None if returner is None else returner.player_id
    kwargs["punter_skill"] = _kick_skill(punter)
    kwargs["return_skill"] = _return_skill(returner)
    return _BASE_PUNT(rng, **kwargs)


def simulate_field_goal_v72(rng, **kwargs):
    if _BASE_FIELD_GOAL is None:
        raise RuntimeError("v7.2 base field-goal hook is not configured")
    kicker = _specialist(_CURRENT_OFFENSE, {"K"})
    kwargs["kicker_id"] = None if kicker is None else kicker.player_id
    kwargs["kicking_skill"] = _kick_skill(kicker)
    return _BASE_FIELD_GOAL(rng, **kwargs)


def simulate_kickoff_v72(rng, **kwargs):
    if _BASE_KICKOFF is None:
        raise RuntimeError("v7.2 base kickoff hook is not configured")
    kicker = _specialist(_CURRENT_OFFENSE, {"K"})
    returner = _returner(_CURRENT_DEFENSE)
    kwargs["kicker_id"] = None if kicker is None else kicker.player_id
    kwargs["returner_id"] = None if returner is None else returner.player_id
    kwargs["return_skill"] = _return_skill(returner)
    return _BASE_KICKOFF(rng, **kwargs)


def capture_special_teams_v72(result: object) -> None:
    game = (
        f"{getattr(result.final_state, 'away_team_id', '')}@"
        f"{getattr(result.final_state, 'home_team_id', '')}"
    )
    for index, event in enumerate(getattr(result, "special_teams_events", ())):
        _EVENT_ROWS.append(
            {
                "game": game,
                "sequence": index,
                "event_type": str(getattr(event, "event_type", "")),
                "kicker_id": getattr(event, "kicker_id", None),
                "punter_id": getattr(event, "punter_id", None),
                "returner_id": getattr(event, "returner_id", None),
                "kick_distance": float(getattr(event, "kick_distance", 0.0) or 0.0),
                "return_yards": float(getattr(event, "return_yards", 0.0) or 0.0),
                "touchback": bool(getattr(event, "touchback", False)),
                "fair_catch": bool(getattr(event, "fair_catch", False)),
                "muffed": bool(getattr(event, "muffed", False)),
                "blocked": bool(getattr(event, "blocked", False)),
                "made": getattr(event, "made", None),
                "return_touchdown": bool(
                    getattr(event, "return_touchdown", False)
                ),
            }
        )


def write_special_teams_telemetry_v72(out: Path) -> None:
    if not _EVENT_ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_EVENT_ROWS).write_csv(
        out / "same_world_special_teams_players_v72.csv"
    )


def player_team_v72(player_id: str | None) -> str:
    if not player_id:
        return ""
    role = role_for(str(player_id))
    return "" if role is None else role.team_id


def kickoff_loop_v72(state, *args, **kwargs):
    """Expose kicking/receiving identity to the inherited kickoff ecology."""

    if _BASE_KICKOFF_LOOP is None:
        raise RuntimeError("v7.2 base kickoff loop is not configured")
    set_kick_context_v72(
        str(getattr(state, "possession", "")),
        str(getattr(state, "defense", "")),
    )
    return _BASE_KICKOFF_LOOP(state, *args, **kwargs)
