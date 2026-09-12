from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class PossessionTerminal(StrEnum):
    TOUCHDOWN = "touchdown"
    DEFENSIVE_TOUCHDOWN = "defensive_touchdown"
    SPECIAL_TEAMS_TOUCHDOWN = "special_teams_touchdown"
    SPECIAL_TEAMS_TURNOVER = "special_teams_turnover"
    FIELD_GOAL = "field_goal"
    MISSED_FIELD_GOAL = "missed_field_goal"
    SAFETY = "safety"
    PUNT = "punt"
    TURNOVER = "turnover"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    HALFTIME = "halftime"
    END_GAME = "end_game"


@dataclass(frozen=True)
class FootballState:
    """Minimal pre-snap state shared by both teams in one simulated world.

    `yardline_100` is yards from the possessing offense's goal line (1..99). Keeping one
    offense-relative coordinate makes down/distance updates simple; change of possession mirrors
    the field with `100 - yardline_100`.

    Live games carry explicit away/home team ids so score-margin decisions remain correct
    after possession changes. Legacy synthetic states that literally use ``away``/``home``
    can omit them.
    """

    possession: str
    defense: str
    quarter: int = 1
    seconds_remaining: int = 3600
    yardline_100: float = 25.0
    down: int = 1
    distance: float = 10.0
    away_score: int = 0
    home_score: int = 0
    away_team_id: str | None = None
    home_team_id: str | None = None

    def __post_init__(self) -> None:
        if self.possession == self.defense:
            raise ValueError("possession and defense must be different teams")
        if not 1 <= self.quarter <= 5:
            raise ValueError("quarter must be 1..5")
        if self.seconds_remaining < 0:
            raise ValueError("seconds_remaining cannot be negative")
        if not 0.0 < self.yardline_100 < 100.0:
            raise ValueError("yardline_100 must be between goal lines")
        if not 1 <= self.down <= 4:
            raise ValueError("down must be 1..4")
        if self.distance <= 0:
            raise ValueError("distance must be positive")
        if (self.away_team_id is None) != (self.home_team_id is None):
            raise ValueError("away_team_id and home_team_id must be supplied together")
        if self.away_team_id is not None and self.away_team_id == self.home_team_id:
            raise ValueError("away_team_id and home_team_id must be different")

    @property
    def score_margin_for_offense(self) -> int:
        if self.away_team_id is not None:
            if self.possession == self.away_team_id:
                return self.away_score - self.home_score
            if self.possession == self.home_team_id:
                return self.home_score - self.away_score
            raise ValueError("possession must match away_team_id or home_team_id")
        if self.possession == "away":
            return self.away_score - self.home_score
        if self.possession == "home":
            return self.home_score - self.away_score
        raise ValueError(
            "non-literal team ids require away_team_id and home_team_id for score-margin decisions"
        )


def mirror_field(yardline_100: float) -> float:
    return float(min(max(100.0 - yardline_100, 1.0), 99.0))


def next_series_distance(yardline_100: float) -> float:
    return float(min(10.0, 100.0 - yardline_100))


def apply_scrimmage_yards(state: FootballState, yards: float, elapsed_seconds: int) -> FootballState:
    """Advance one non-turnover scrimmage play through field/down/distance/clock state."""
    elapsed = max(int(elapsed_seconds), 0)
    new_clock = max(state.seconds_remaining - elapsed, 0)
    new_yardline = state.yardline_100 + float(yards)
    if new_yardline >= 100.0:
        # Scoring transition is handled by the possession terminal layer; clamp the snap state.
        new_yardline = 99.999
    else:
        new_yardline = max(new_yardline, 1.0)

    gained_first = yards >= state.distance
    if gained_first:
        down = 1
        distance = next_series_distance(new_yardline)
    else:
        down = min(state.down + 1, 4)
        distance = max(state.distance - float(yards), 1.0)
    return replace(
        state,
        seconds_remaining=new_clock,
        yardline_100=new_yardline,
        down=down,
        distance=distance,
    )


def change_possession(
    state: FootballState,
    *,
    receiving_yardline_100: float,
    elapsed_seconds: int = 0,
) -> FootballState:
    """Hand the same physical field state to the opponent rather than resetting to the 25."""
    elapsed = max(int(elapsed_seconds), 0)
    receiving = float(min(max(receiving_yardline_100, 1.0), 99.0))
    return FootballState(
        possession=state.defense,
        defense=state.possession,
        quarter=state.quarter,
        seconds_remaining=max(state.seconds_remaining - elapsed, 0),
        yardline_100=receiving,
        down=1,
        distance=next_series_distance(receiving),
        away_score=state.away_score,
        home_score=state.home_score,
        away_team_id=state.away_team_id,
        home_team_id=state.home_team_id,
    )


def turnover_at_spot(
    state: FootballState, spot_yardline_100: float, elapsed_seconds: int = 0
) -> FootballState:
    """A turnover gives the opponent the mirrored physical spot."""
    return change_possession(
        state,
        receiving_yardline_100=mirror_field(spot_yardline_100),
        elapsed_seconds=elapsed_seconds,
    )


def turnover_on_downs(state: FootballState, elapsed_seconds: int = 0) -> FootballState:
    return turnover_at_spot(state, state.yardline_100, elapsed_seconds)


def punt_transition(
    state: FootballState,
    *,
    gross_yards: float,
    return_yards: float = 0.0,
    touchback: bool = False,
    elapsed_seconds: int = 8,
) -> FootballState:
    """Resolve punt field position into the opponent's offense-relative coordinate."""
    if touchback or state.yardline_100 + gross_yards >= 100.0:
        receiving = 20.0
    else:
        physical_end = state.yardline_100 + max(float(gross_yards), 0.0)
        receiving = mirror_field(physical_end) + max(float(return_yards), 0.0)
        receiving = min(max(receiving, 1.0), 99.0)
    return change_possession(
        state,
        receiving_yardline_100=receiving,
        elapsed_seconds=elapsed_seconds,
    )


def kickoff_transition(
    state: FootballState,
    *,
    receiving_yardline_100: float = 35.0,
    elapsed_seconds: int = 6,
) -> FootballState:
    return change_possession(
        state,
        receiving_yardline_100=receiving_yardline_100,
        elapsed_seconds=elapsed_seconds,
    )


def missed_field_goal_transition(
    state: FootballState,
    *,
    kick_spot_yardline_100: float | None = None,
    elapsed_seconds: int = 5,
) -> FootballState:
    """Missed FG hands over field position at the kick spot, bounded by touchback-style field state."""
    spot = state.yardline_100 if kick_spot_yardline_100 is None else kick_spot_yardline_100
    receiving = mirror_field(spot)
    return change_possession(
        state,
        receiving_yardline_100=max(receiving, 20.0),
        elapsed_seconds=elapsed_seconds,
    )
