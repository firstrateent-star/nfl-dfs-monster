from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from monster.reality.world_state import PregameWorld, TeamPregameState


class ExitReason(StrEnum):
    INJURY = "injury"
    CONCUSSION = "concussion"
    BENCH = "bench"
    ROTATION = "rotation"
    EJECTION = "ejection"
    OTHER = "other"


@dataclass(frozen=True)
class AvailabilityTransition:
    team_id: str
    player_id: str
    reason: ExitReason
    quarter: int
    seconds_remaining: int
    replacement_player_id: str | None = None


@dataclass(frozen=True)
class TeamLiveRoster:
    """Live game roster state derived from one immutable pregame roster."""

    team_id: str
    active_player_ids: tuple[str, ...]
    current_qb_id: str | None
    exited_player_ids: tuple[str, ...] = ()

    @classmethod
    def from_pregame(cls, team: TeamPregameState) -> TeamLiveRoster:
        return cls(
            team_id=team.team_id,
            active_player_ids=team.active_player_ids,
            current_qb_id=team.starting_qb_id,
        )

    def __post_init__(self) -> None:
        if len(self.active_player_ids) != len(set(self.active_player_ids)):
            raise ValueError("live active player ids must be unique")
        if len(self.exited_player_ids) != len(set(self.exited_player_ids)):
            raise ValueError("exited player ids must be unique")
        if set(self.active_player_ids) & set(self.exited_player_ids):
            raise ValueError("a player cannot be active and exited simultaneously")
        if self.current_qb_id is not None and self.current_qb_id not in self.active_player_ids:
            raise ValueError("current QB must be active")

    def exit_player(self, player_id: str) -> TeamLiveRoster:
        if player_id not in self.active_player_ids:
            raise ValueError("only an active player can exit")
        remaining = tuple(pid for pid in self.active_player_ids if pid != player_id)
        exited = (*self.exited_player_ids, player_id)
        return replace(
            self,
            active_player_ids=remaining,
            exited_player_ids=exited,
            current_qb_id=None if self.current_qb_id == player_id else self.current_qb_id,
        )

    def activate_qb(self, player_id: str) -> TeamLiveRoster:
        if player_id not in self.active_player_ids:
            raise ValueError("replacement QB must already be active in the game world")
        return replace(self, current_qb_id=player_id)


@dataclass(frozen=True)
class GameAvailabilityState:
    away: TeamLiveRoster
    home: TeamLiveRoster
    transitions: tuple[AvailabilityTransition, ...] = ()

    @classmethod
    def from_pregame(cls, world: PregameWorld) -> GameAvailabilityState:
        return cls(
            away=TeamLiveRoster.from_pregame(world.away),
            home=TeamLiveRoster.from_pregame(world.home),
        )

    def team(self, team_id: str) -> TeamLiveRoster:
        if team_id == self.away.team_id:
            return self.away
        if team_id == self.home.team_id:
            return self.home
        raise ValueError("team_id is not part of this game")

    def exit_player(
        self,
        *,
        team_id: str,
        player_id: str,
        reason: ExitReason,
        quarter: int,
        seconds_remaining: int,
        replacement_qb_id: str | None = None,
    ) -> GameAvailabilityState:
        team = self.team(team_id)
        was_current_qb = team.current_qb_id == player_id
        updated = team.exit_player(player_id)
        if replacement_qb_id is not None:
            if not was_current_qb:
                raise ValueError("replacement_qb_id is only valid when the current QB exits")
            updated = updated.activate_qb(replacement_qb_id)

        transition = AvailabilityTransition(
            team_id=team_id,
            player_id=player_id,
            reason=reason,
            quarter=quarter,
            seconds_remaining=seconds_remaining,
            replacement_player_id=replacement_qb_id,
        )
        if team_id == self.away.team_id:
            return replace(
                self,
                away=updated,
                transitions=(*self.transitions, transition),
            )
        return replace(
            self,
            home=updated,
            transitions=(*self.transitions, transition),
        )
