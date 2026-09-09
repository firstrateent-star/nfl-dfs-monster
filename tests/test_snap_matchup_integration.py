from __future__ import annotations

import numpy as np

from monster.sim.football_state import FootballState
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity, simulate_scrimmage_play


def _offense() -> TeamIdentity:
    qb = PlayerIdentity("qb", "QB", "QB", usage_weight=0.05)
    rb = PlayerIdentity("rb", "RB", "RB", usage_weight=1.0)
    wr = PlayerIdentity("wr", "WR", "WR", usage_weight=1.0)
    return TeamIdentity("OFF", qb, (rb,), (wr,), neutral_pass_rate=1.0)


def _defense() -> DefensiveUnit:
    edge = DefensiveIdentity("edge", "EDGE", "EDGE", pass_rush=1.2, run_defense=1.2)
    cb = DefensiveIdentity("cb", "CB", "CB", coverage=1.15, ball_hawk=1.15)
    return DefensiveUnit(front=(edge,), coverage=(cb,))


def _state() -> FootballState:
    return FootballState(
        possession="OFF",
        defense="DEF",
        down=1,
        distance=10.0,
        yardline_100=25.0,
        away_team_id="OFF",
        home_team_id="DEF",
    )


def test_pass_snap_carries_real_matchup_defender_identity() -> None:
    state = _state()
    rng = np.random.default_rng(91)
    events = [simulate_scrimmage_play(state, _offense(), 1.0, rng, defense=_defense()) for _ in range(40)]
    pass_events = [event for event in events if event.play_type == "pass"]
    assert pass_events
    assert all(event.primary_defender_id == "cb" for event in pass_events if event.target_id is not None)


def test_run_snap_records_front_interaction_and_stuff_state() -> None:
    offense = _offense()
    offense = TeamIdentity(
        offense.team_id,
        offense.quarterback,
        offense.rushers,
        offense.receivers,
        neutral_pass_rate=0.0,
        run_blocking=0.75,
    )
    state = _state()
    rng = np.random.default_rng(92)
    events = [simulate_scrimmage_play(state, offense, 1.0, rng, defense=_defense()) for _ in range(80)]
    run_events = [event for event in events if event.play_type == "run"]
    assert run_events
    assert all(event.primary_defender_id == "edge" for event in run_events)
    assert any(event.stuffed for event in run_events)


def test_matchup_path_preserves_event_player_identity() -> None:
    state = _state()
    rng = np.random.default_rng(93)
    events = [simulate_scrimmage_play(state, _offense(), 1.0, rng, defense=_defense()) for _ in range(30)]
    assert all(event.passer_id == "qb" for event in events if event.play_type == "pass")
    assert all(event.target_id in {"wr", None} for event in events if event.play_type == "pass")
