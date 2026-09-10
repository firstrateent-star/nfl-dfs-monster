import polars as pl

from monster.feature_compile.game_flow_policy import compile_game_flow_policy
from monster.sim.football_state import FootballState
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_lookup import build_team_game_flow_policy, context_key


def _pbp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "posteam": ["A", "A", "B", "B", "B", "A"],
            "qb_dropback": [1, 0, 1, 1, 0, 1],
            "rush_attempt": [0, 1, 0, 0, 1, 0],
            "down": [3, 3, 3, 3, 3, 2],
            "ydstogo": [10, 10, 10, 10, 10, 2],
            "yardline_100": [50, 50, 50, 50, 50, 75],
            "qtr": [1, 1, 1, 1, 1, 1],
            "game_seconds_remaining": [3300, 3280, 3200, 3180, 3160, 3100],
            "score_differential": [0, 0, 0, 0, 0, 0],
        }
    )


def test_compiler_preserves_raw_team_samples() -> None:
    league, team = compile_game_flow_policy(_pbp())
    key_rows = team.filter(
        (pl.col("team_id") == "A")
        & (pl.col("down") == 3)
        & (pl.col("distance_bucket") == "8_10")
    )
    assert key_rows.height == 1
    assert key_rows.item(0, "samples") == 2
    assert key_rows.item(0, "dropback_rate") == 0.5
    assert league.get_column("samples").sum() == 6


def test_runtime_key_matches_historical_coordinate_semantics() -> None:
    flow = derive_game_flow_state(
        FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
            down=3,
            distance=10.0,
            yardline_100=50.0,
        )
    )
    key = context_key(flow)
    assert key.distance_bucket == "8_10"
    assert key.field_zone == "midfield"
    assert key.time_mode == "q1_normal"
    assert key.score_state == "tied"


def test_lookup_shrinks_sparse_team_cell_to_matching_league_state() -> None:
    league, team = compile_game_flow_policy(_pbp())
    policy = build_team_game_flow_policy(
        team_id="A",
        league_rows=league.to_dicts(),
        team_rows=team.to_dicts(),
        team_neutral_rate=0.50,
        league_neutral_rate=0.60,
        shrinkage_samples=80.0,
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
        )
    )
    evidence = policy.evidence_for(flow)
    assert evidence.league_context.samples == 5
    assert evidence.team_context is not None
    assert evidence.team_context.samples == 2
    assert evidence.team_context.rate == 0.5
