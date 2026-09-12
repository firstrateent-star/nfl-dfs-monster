from __future__ import annotations

import numpy as np

from monster.sim.chaos_ecology import ChaosEcology, ReturnKind, resolve_turnover_return
from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.football_state import FootballState
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType, PlayerIdentity, TeamIdentity
from monster.sim.special_teams_v13 import simulate_kickoff, simulate_punt


def _defense() -> DefensiveUnit:
    defender = DefensiveIdentity(
        player_id="D1",
        name="Defender",
        position="CB",
        coverage=1.1,
        ball_hawk=1.1,
        speed=1.15,
        returning=1.2,
    )
    return DefensiveUnit(front=(), coverage=(defender,))


def _interception(*, air_yards: float = 0.0) -> PlayEvent:
    return PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="QB",
        target_id="WR",
        primary_defender_id="D1",
        pass_result=PassResult.INTERCEPTION,
        turnover=True,
        air_yards=air_yards,
    )


def test_interception_change_spot_uses_throw_geometry() -> None:
    before = FootballState(
        possession="A",
        defense="B",
        yardline_100=40.0,
        away_team_id="A",
        home_team_id="B",
    )
    ecology = ChaosEcology(interception_zero_return_rate=1.0)
    result = resolve_turnover_return(
        before,
        _interception(air_yards=20.0),
        defense=_defense(),
        rng=np.random.default_rng(7),
        ecology=ecology,
    )
    assert result.kind == ReturnKind.INTERCEPTION
    assert result.change_spot_yardline_100 == 60.0
    assert result.return_yards == 0.0
    assert result.receiving_yardline_100 == 40.0
    assert not result.touchdown


def test_end_zone_interception_is_touchback() -> None:
    before = FootballState(
        possession="A",
        defense="B",
        yardline_100=91.0,
        away_team_id="A",
        home_team_id="B",
    )
    result = resolve_turnover_return(
        before,
        _interception(air_yards=15.0),
        defense=_defense(),
        rng=np.random.default_rng(9),
    )
    assert result.touchback
    assert result.receiving_yardline_100 == 20.0
    assert not result.touchdown


def test_return_touchdown_requires_covering_remaining_field() -> None:
    before = FootballState(
        possession="A",
        defense="B",
        yardline_100=10.0,
        away_team_id="A",
        home_team_id="B",
    )
    ecology = ChaosEcology(
        interception_zero_return_rate=0.0,
        interception_return_mean=50.0,
        interception_return_sd=1.0,
        interception_40_plus_rate=1.0,
    )
    result = resolve_turnover_return(
        before,
        _interception(),
        defense=_defense(),
        rng=np.random.default_rng(4),
        ecology=ecology,
    )
    # Interception at A's 10 becomes B ball at B's 90: only 10 return yards are needed.
    assert result.return_yards == 10.0
    assert result.touchdown
    assert result.receiving_yardline_100 == 100.0


def test_special_teams_expose_touchbacks_muffs_and_return_tails() -> None:
    touchback = simulate_kickoff(
        np.random.default_rng(1),
        ecology=ChaosEcology(kickoff_touchback_rate=1.0),
    )
    assert touchback.touchback

    muff = simulate_punt(
        np.random.default_rng(3),
        ecology=ChaosEcology(
            blocked_punt_rate=0.0,
            punt_muff_rate=1.0,
            punt_muff_kicking_recovery_rate=1.0,
        ),
    )
    assert muff.muffed
    assert muff.kicking_team_recovery


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-QB", "QB", "QB", usage_weight=1.0)
    rb = PlayerIdentity(f"{team}-RB", "RB", "RB", usage_weight=1.0)
    wr = PlayerIdentity(f"{team}-WR", "WR", "WR", usage_weight=1.0)
    return TeamIdentity(team_id=team, quarterback=qb, rushers=(rb,), receivers=(wr,))


def test_complete_game_conserves_score_with_chaos_ecology() -> None:
    result = simulate_game(_team("A"), _team("B"), seed=20260912)
    assert_event_conservation(result)
    summary = summarize_game(result)
    assert summary.touchdowns == (
        summary.offensive_touchdowns
        + summary.defensive_touchdowns
        + summary.special_teams_touchdowns
    )
    assert summary.drive_start_yardline_mean > 0.0
    assert summary.longest_scrimmage_play >= 0.0
