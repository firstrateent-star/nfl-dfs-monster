from __future__ import annotations

import monster.sim.game_loop_v13 as game_loop
from monster.sim.clock import OVERTIME_SECONDS, advance_game_clock
from monster.sim.decision_policy import FourthDownDecision, fourth_down_decision
from monster.sim.event_ledger import assert_event_conservation
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import PlayerIdentity, PlayEvent, PlayType, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.12)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.75)
    wr = PlayerIdentity(f"{team}-wr", f"{team} WR", "WR", usage_weight=1.0)
    return TeamIdentity(team, qb, (rb, qb), (wr,))


def test_overtime_clock_stays_in_fifth_period() -> None:
    state = FootballState(
        possession="away",
        defense="home",
        quarter=5,
        seconds_remaining=OVERTIME_SECONDS,
        away_team_id="away",
        home_team_id="home",
    )
    after = advance_game_clock(state, 31)
    assert after.quarter == 5
    assert after.seconds_remaining == OVERTIME_SECONDS - 31


def test_trailing_overtime_response_drive_cannot_punt_away_game() -> None:
    own_territory = FootballState(
        possession="away",
        defense="home",
        quarter=5,
        seconds_remaining=500,
        yardline_100=35.0,
        down=4,
        distance=12.0,
        away_score=20,
        home_score=27,
        away_team_id="away",
        home_team_id="home",
    )
    field_goal_range = FootballState(
        possession="away",
        defense="home",
        quarter=5,
        seconds_remaining=500,
        yardline_100=70.0,
        down=4,
        distance=5.0,
        away_score=20,
        home_score=23,
        away_team_id="away",
        home_team_id="home",
    )
    assert fourth_down_decision(own_territory) == FourthDownDecision.GO
    assert fourth_down_decision(field_goal_range) == FourthDownDecision.FIELD_GOAL


def test_complete_game_enters_overtime_from_tied_regulation(monkeypatch) -> None:
    tied_regulation = game_loop.GameResultV13(
        final_state=FootballState(
            possession="away",
            defense="home",
            quarter=4,
            seconds_remaining=0,
            away_score=0,
            home_score=0,
            away_team_id="away",
            home_team_id="home",
        ),
        plays=(),
        player_stats={},
        drives=0,
    )

    monkeypatch.setattr(
        game_loop,
        "simulate_regulation_game",
        lambda *args, **kwargs: tied_regulation,
    )
    result = game_loop.simulate_game(_team("away"), _team("home"), seed=17)

    assert result.went_to_overtime is True
    assert result.final_state.quarter == 5
    assert 0 <= result.final_state.seconds_remaining <= OVERTIME_SECONDS
    assert_event_conservation(result)


def test_game_ending_overtime_touchdown_conserves_score_without_try() -> None:
    touchdown = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=8,
        yards=12.0,
        rusher_id="away-rb",
        touchdown=True,
    )
    result = game_loop.GameResultV13(
        final_state=FootballState(
            possession="away",
            defense="home",
            quarter=5,
            seconds_remaining=240,
            away_score=6,
            home_score=0,
            away_team_id="away",
            home_team_id="home",
        ),
        plays=(touchdown,),
        player_stats={
            "away-rb": game_loop.PlayerBoxScore(
                rush_attempts=1,
                rushing_yards=12.0,
                rushing_tds=1,
            )
        },
        drives=1,
        went_to_overtime=True,
        overtime_touchdowns_without_try=1,
    )
    assert_event_conservation(result)
