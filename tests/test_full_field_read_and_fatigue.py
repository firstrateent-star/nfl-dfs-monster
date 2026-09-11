import numpy as np

from monster.sim.football_state import FootballState
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity, simulate_scrimmage_play


def _defense() -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            DefensiveIdentity("edge", "EDGE", "EDGE", pass_rush=1.08, run_defense=1.0, tackling=1.0),
        ),
        coverage=(
            DefensiveIdentity("cb1", "CB1", "CB", coverage=1.12, ball_hawk=1.04, snap_weight=1.0),
            DefensiveIdentity("cb2", "CB2", "CB", coverage=0.86, ball_hawk=0.92, snap_weight=0.9),
            DefensiveIdentity("fs", "FS", "FS", coverage=1.02, ball_hawk=1.06, snap_weight=0.95),
        ),
    )


def _team() -> TeamIdentity:
    qb = PlayerIdentity("qb", "QB", "QB", efficiency=1.05, explosive=0.95)
    covered = PlayerIdentity("wr1", "Covered", "WR", usage_weight=0.34, efficiency=0.88, explosive=1.0)
    open_receiver = PlayerIdentity("wr2", "Open", "WR", usage_weight=0.30, efficiency=1.18, explosive=1.05)
    checkdown = PlayerIdentity("rb", "Back", "RB", usage_weight=0.16, efficiency=1.0, explosive=0.95)
    return TeamIdentity(
        "OFF",
        quarterback=qb,
        rushers=(checkdown,),
        receivers=(covered, open_receiver, checkdown),
        neutral_pass_rate=1.0,
        pass_efficiency=1.04,
    )


def test_full_field_read_can_redirect_from_usage_leader() -> None:
    offense = _team()
    defense = _defense()
    counts = {player.player_id: 0 for player in offense.receivers}
    for seed in range(250):
        state = FootballState(possession="OFF", defense="DEF", away_team_id="OFF", home_team_id="DEF")
        event = simulate_scrimmage_play(state, offense, 1.0, np.random.default_rng(seed), defense)
        if event.target_id in counts:
            counts[event.target_id] += 1
    assert counts["wr2"] > counts["wr1"]


def test_fatigue_is_world_local_and_accumulates() -> None:
    offense = _team()
    defense = _defense()
    rng = np.random.default_rng(991)
    state = FootballState(possession="OFF", defense="DEF", away_team_id="OFF", home_team_id="DEF")
    factors = []
    for play in range(55):
        state = FootballState(
            possession="OFF",
            defense="DEF",
            away_team_id="OFF",
            home_team_id="DEF",
            down=1 + play % 3,
            seconds_remaining=3600 - play * 40,
        )
        event = simulate_scrimmage_play(state, offense, 1.0, rng, defense)
        factors.append(event.fatigue_factor)
    assert min(factors[-10:]) <= factors[0]

    fresh = simulate_scrimmage_play(
        FootballState(possession="OFF", defense="DEF", away_team_id="OFF", home_team_id="DEF"),
        offense,
        1.0,
        np.random.default_rng(992),
        defense,
    )
    assert fresh.fatigue_factor == 1.0
