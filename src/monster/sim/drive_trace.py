from __future__ import annotations

from dataclasses import dataclass

from monster.sim.football_state import FootballState, PossessionTerminal
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


@dataclass(frozen=True)
class DriveTrace:
    """Observed anatomy of one simulated possession.

    This is an audit record, not a scoring prior. It describes what the event engine did so
    drive conversion can be compared with historical football before any mechanism changes.
    ``red_zone_entered`` means the possession reached the red zone, including on a scoring
    play from outside it. ``red_zone_snap_seen`` is stricter: the offense actually began an
    observed event with the ball in the red zone.

    Survival fields are intentionally scrimmage-defined. They localize where possessions
    live or die without giving the audit any authority over play selection or resolution.
    """

    offense_team_id: str
    defense_team_id: str
    start_quarter: int
    start_seconds_remaining: int
    start_yardline_100: float
    start_score_margin: int
    end_quarter: int
    end_seconds_remaining: int
    end_yardline_100: float
    terminal: PossessionTerminal
    points: int
    scrimmage_plays: int
    net_scrimmage_yards: float
    first_downs: int
    explosive_plays: int
    red_zone_entered: bool
    red_zone_snap_seen: bool
    goal_to_go_reached: bool
    goal_to_go_snap_seen: bool
    pressured_dropbacks: int
    sacks: int
    turnovers: int
    overtime: bool
    series_started: int
    series_converted: int
    first_down_snaps: int
    second_down_snaps: int
    third_down_snaps: int
    fourth_down_snaps: int
    third_down_conversions: int
    third_and_long_snaps: int
    third_and_long_conversions: int
    third_down_distance_total: float
    early_down_5plus_gains: int


class DriveTraceRecorder:
    """Passive possession observer with no authority over simulation behavior."""

    def __init__(self, start: FootballState) -> None:
        self.start = start
        self.scrimmage_plays = 0
        self.net_scrimmage_yards = 0.0
        self.first_downs = 0
        self.explosive_plays = 0
        self.red_zone_entered = start.yardline_100 >= 80.0
        self.red_zone_snap_seen = start.yardline_100 >= 80.0
        self.goal_to_go_reached = _goal_to_go(start)
        self.goal_to_go_snap_seen = _goal_to_go(start)
        self.pressured_dropbacks = 0
        self.sacks = 0
        self.turnovers = 0
        self.last_offense_yardline = start.yardline_100
        self.observed_events = 0
        self.series_started = 0
        self.series_converted = 0
        self.first_down_snaps = 0
        self.second_down_snaps = 0
        self.third_down_snaps = 0
        self.fourth_down_snaps = 0
        self.third_down_conversions = 0
        self.third_and_long_snaps = 0
        self.third_and_long_conversions = 0
        self.third_down_distance_total = 0.0
        self.early_down_5plus_gains = 0

    def observe(self, before: FootballState, event: PlayEvent) -> None:
        """Observe a resolved event while it still belongs to the current offense."""
        if before.possession != self.start.possession:
            raise ValueError("drive observer received an event from a different offense")
        self.observed_events += 1
        self.red_zone_snap_seen = self.red_zone_snap_seen or before.yardline_100 >= 80.0
        self.goal_to_go_snap_seen = self.goal_to_go_snap_seen or _goal_to_go(before)
        if event.play_type not in {PlayType.RUN, PlayType.PASS}:
            return

        self.scrimmage_plays += 1
        self.net_scrimmage_yards += float(event.yards)
        end_yardline = min(max(before.yardline_100 + float(event.yards), 1.0), 100.0)
        self.last_offense_yardline = end_yardline
        self.red_zone_entered = self.red_zone_entered or end_yardline >= 80.0
        self.goal_to_go_reached = self.goal_to_go_reached or _goal_to_go(before)
        self.explosive_plays += int(float(event.yards) >= 15.0)
        self.pressured_dropbacks += int(event.play_type == PlayType.PASS and event.pressured)
        self.sacks += int(event.pass_result == PassResult.SACK)
        self.turnovers += int(event.turnover)

        if before.down == 1:
            self.series_started += 1
            self.first_down_snaps += 1
        elif before.down == 2:
            self.second_down_snaps += 1
        elif before.down == 3:
            self.third_down_snaps += 1
            self.third_down_distance_total += float(before.distance)
            if before.distance >= 7.0:
                self.third_and_long_snaps += 1
        elif before.down == 4:
            self.fourth_down_snaps += 1

        converted = (
            not event.touchdown
            and not event.turnover
            and float(event.yards) >= float(before.distance)
        )
        if converted:
            self.first_downs += 1
            self.series_converted += 1
            if before.down == 3:
                self.third_down_conversions += 1
                if before.distance >= 7.0:
                    self.third_and_long_conversions += 1

        if before.down in {1, 2} and float(event.yards) >= 5.0:
            self.early_down_5plus_gains += 1

    @property
    def has_activity(self) -> bool:
        return self.observed_events > 0

    def observe_penalty(self, before: FootballState, after: FootballState) -> None:
        """Observe an enforced penalty without counting it as a scrimmage play."""
        if before.possession != self.start.possession:
            raise ValueError("drive observer received a penalty from a different offense")
        self.observed_events += 1
        self.red_zone_snap_seen = self.red_zone_snap_seen or before.yardline_100 >= 80.0
        self.goal_to_go_snap_seen = self.goal_to_go_snap_seen or _goal_to_go(before)
        if after.possession == self.start.possession:
            self.last_offense_yardline = after.yardline_100
            self.red_zone_entered = self.red_zone_entered or after.yardline_100 >= 80.0
            self.goal_to_go_reached = self.goal_to_go_reached or _goal_to_go(after)

    def finish(
        self,
        end: FootballState,
        terminal: PossessionTerminal,
        *,
        points: int | None = None,
    ) -> DriveTrace:
        if points is None:
            points = _score_for_team(end, self.start.possession) - _score_for_team(
                self.start, self.start.possession
            )
        return DriveTrace(
            offense_team_id=self.start.possession,
            defense_team_id=self.start.defense,
            start_quarter=self.start.quarter,
            start_seconds_remaining=self.start.seconds_remaining,
            start_yardline_100=self.start.yardline_100,
            start_score_margin=self.start.score_margin_for_offense,
            end_quarter=end.quarter,
            end_seconds_remaining=end.seconds_remaining,
            end_yardline_100=float(self.last_offense_yardline),
            terminal=terminal,
            points=int(points),
            scrimmage_plays=self.scrimmage_plays,
            net_scrimmage_yards=float(self.net_scrimmage_yards),
            first_downs=self.first_downs,
            explosive_plays=self.explosive_plays,
            red_zone_entered=self.red_zone_entered,
            red_zone_snap_seen=self.red_zone_snap_seen,
            goal_to_go_reached=self.goal_to_go_reached,
            goal_to_go_snap_seen=self.goal_to_go_snap_seen,
            pressured_dropbacks=self.pressured_dropbacks,
            sacks=self.sacks,
            turnovers=self.turnovers,
            overtime=self.start.quarter == 5,
            series_started=self.series_started,
            series_converted=self.series_converted,
            first_down_snaps=self.first_down_snaps,
            second_down_snaps=self.second_down_snaps,
            third_down_snaps=self.third_down_snaps,
            fourth_down_snaps=self.fourth_down_snaps,
            third_down_conversions=self.third_down_conversions,
            third_and_long_snaps=self.third_and_long_snaps,
            third_and_long_conversions=self.third_and_long_conversions,
            third_down_distance_total=float(self.third_down_distance_total),
            early_down_5plus_gains=self.early_down_5plus_gains,
        )


def _goal_to_go(state: FootballState) -> bool:
    yards_to_goal = 100.0 - state.yardline_100
    return yards_to_goal <= 10.0 and state.distance >= yards_to_goal - 1e-9


def _score_for_team(state: FootballState, team_id: str) -> int:
    if state.away_team_id is not None:
        if team_id == state.away_team_id:
            return state.away_score
        if team_id == state.home_team_id:
            return state.home_score
        raise ValueError("team_id must match away or home team")
    if team_id == "away":
        return state.away_score
    if team_id == "home":
        return state.home_score
    raise ValueError("explicit away/home ids required for non-literal team ids")
