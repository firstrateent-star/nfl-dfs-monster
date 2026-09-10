from monster.sim.football_state import FootballState
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.play_kernel import PlayType, PlayerIdentity, TeamIdentity, choose_play_type


class _FixedRng:
    def random(self) -> float:
        return 0.50


def _team(policy) -> TeamIdentity:
    qb = PlayerIdentity("qb", "QB", "QB")
    rb = PlayerIdentity("rb", "RB", "RB")
    wr = PlayerIdentity("wr", "WR", "WR")
    return TeamIdentity(
        team_id="A",
        quarterback=qb,
        rushers=(rb,),
        receivers=(wr,),
        neutral_pass_rate=0.20,
        league_neutral_pass_rate=0.60,
        game_flow_policy=policy,
    )


def _policy() -> object:
    row = {
        "down": 3,
        "distance_bucket": "8_10",
        "field_zone": "midfield",
        "time_mode": "q1_normal",
        "score_state": "tied",
        "samples": 1000,
        "dropback_rate": 0.99,
    }
    return build_team_game_flow_policy(
        team_id="A",
        league_rows=[row],
        team_rows=[],
        team_neutral_rate=None,
        league_neutral_rate=None,
    )


def test_game_flow_policy_can_own_normal_run_dropback_choice() -> None:
    state = FootballState(
        possession="A",
        defense="B",
        away_team_id="A",
        home_team_id="B",
        down=3,
        distance=10.0,
        yardline_100=50.0,
    )
    assert choose_play_type(state, _team(_policy()), _FixedRng()) == PlayType.PASS


def test_game_flow_policy_does_not_override_fourth_down_punt_boundary() -> None:
    state = FootballState(
        possession="A",
        defense="B",
        away_team_id="A",
        home_team_id="B",
        down=4,
        distance=10.0,
        yardline_100=30.0,
    )
    assert choose_play_type(state, _team(_policy()), _FixedRng()) == PlayType.PUNT
