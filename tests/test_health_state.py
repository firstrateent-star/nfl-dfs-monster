import polars as pl

from monster.feature_compile.health import attach_health_state
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.snapshot.player import PlayerState, TeamPlayerPool


def test_questionable_health_is_separate_from_role():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["p1"],
            "team_id": ["BUF"],
            "display_name": ["Player One"],
        }
    )
    injuries = pl.DataFrame(
        {
            "gsis_id": ["p1"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation"],
            "report_primary_injury": ["ankle"],
            "week": [1],
        }
    )
    out = attach_health_state(personnel, injuries)
    row = out.to_dicts()[0]
    assert row["health_availability_probability"] == 0.68
    assert row["health_effectiveness_if_active"] == 0.96
    assert row["health_uncertainty"] == 0.22
    assert row["health_state"] == "questionable"


def test_missing_health_evidence_is_neutral():
    personnel = pl.DataFrame(
        {"gsis_id": ["p1"], "team_id": ["BUF"], "display_name": ["Player One"]}
    )
    out = attach_health_state(personnel, pl.DataFrame())
    row = out.to_dicts()[0]
    assert row["health_availability_probability"] == 1.0
    assert row["health_effectiveness_if_active"] == 1.0
    assert row["health_evidence"] == "roster_only"


def test_health_pool_adapter_overwrites_generic_availability_and_effectiveness():
    pool = TeamPlayerPool(
        team_id="BUF",
        players=(PlayerState("p1", "Player One", "WR", "BUF", active_probability=0.99),),
    )
    personnel = pl.DataFrame(
        {
            "gsis_id": ["p1"],
            "game_day_active_probability": [0.61],
            "health_effectiveness_if_active": [0.93],
        }
    )
    out = apply_health_to_skill_pools({"BUF": pool}, personnel)
    player = out["BUF"].players[0]
    assert player.active_probability == 0.61
    assert player.effectiveness_if_active == 0.93
