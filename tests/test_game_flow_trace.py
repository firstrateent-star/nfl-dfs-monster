from monster.sim.football_state import FootballState
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_trace import GameFlowTraceRecorder, primary_flow_state_label


def test_primary_flow_state_uses_stable_third_down_priority() -> None:
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
    assert primary_flow_state_label(flow) == "third_and_7_plus"


def test_trace_aggregates_without_storing_play_rows() -> None:
    flow = derive_game_flow_state(
        FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
            down=2,
            distance=2.0,
            yardline_100=50.0,
        )
    )
    recorder = GameFlowTraceRecorder()
    recorder.observe(
        game="A@B",
        offense="A",
        flow=flow,
        is_dropback=True,
        predicted_probability=0.60,
        league_prior=0.55,
    )
    recorder.observe(
        game="A@B",
        offense="A",
        flow=flow,
        is_dropback=False,
        predicted_probability=0.40,
        league_prior=0.45,
    )

    rows = recorder.rows()
    aggregate = next(row for row in rows if row["game"] == "ALL")
    assert aggregate["samples"] == 2
    assert aggregate["dropbacks"] == 1
    assert aggregate["simulated_dropback_rate"] == 0.5
    assert aggregate["mean_brain_probability"] == 0.5
    assert aggregate["mean_stabilized_league_prior"] == 0.5
    assert aggregate["sampler_calibration_error"] == 0.0
