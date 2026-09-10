from monster.sim.football_state import FootballState
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_lookup import build_team_game_flow_policy


def test_sparse_exact_league_cell_shrinks_to_broader_parent() -> None:
    league_rows = [
        {
            "down": 3,
            "distance_bucket": "8_10",
            "field_zone": "midfield",
            "time_mode": "q1_normal",
            "score_state": "tied",
            "samples": 1,
            "dropback_rate": 1.0,
        },
        {
            "down": 3,
            "distance_bucket": "8_10",
            "field_zone": "midfield",
            "time_mode": "q1_normal",
            "score_state": "lead_1_8",
            "samples": 99,
            "dropback_rate": 0.5,
        },
    ]
    policy = build_team_game_flow_policy(
        team_id="A",
        league_rows=league_rows,
        team_rows=[],
        team_neutral_rate=None,
        league_neutral_rate=None,
        league_shrinkage_samples=80.0,
    )
    flow = derive_game_flow_state(
        FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
            down=3,
            distance=10.0,
            yardline_100=50.0,
            away_score=0,
            home_score=0,
        )
    )

    evidence = policy.evidence_for(flow)

    assert 0.50 < evidence.league_context.rate < 0.55
    assert evidence.league_context.samples == 1


def test_large_exact_league_cell_keeps_most_of_its_signal() -> None:
    league_rows = [
        {
            "down": 3,
            "distance_bucket": "8_10",
            "field_zone": "midfield",
            "time_mode": "q1_normal",
            "score_state": "tied",
            "samples": 800,
            "dropback_rate": 0.9,
        },
        {
            "down": 3,
            "distance_bucket": "8_10",
            "field_zone": "midfield",
            "time_mode": "q1_normal",
            "score_state": "lead_1_8",
            "samples": 800,
            "dropback_rate": 0.5,
        },
    ]
    policy = build_team_game_flow_policy(
        team_id="A",
        league_rows=league_rows,
        team_rows=[],
        team_neutral_rate=None,
        league_neutral_rate=None,
        league_shrinkage_samples=80.0,
    )
    flow = derive_game_flow_state(
        FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
            down=3,
            distance=10.0,
            yardline_100=50.0,
            away_score=0,
            home_score=0,
        )
    )

    evidence = policy.evidence_for(flow)

    assert evidence.league_context.rate > 0.84
