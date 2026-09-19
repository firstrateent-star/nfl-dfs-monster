import polars as pl

from monster.feature_compile.health import attach_health_state
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.skill_pools import compile_current_skill_pools
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


def test_unavailable_depth_qb_cedes_passing_authority_to_active_backup():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["qb1", "qb2", "qb3"],
            "pfr_id": [None, None, None],
            "team_id": ["ATL", "ATL", "ATL"],
            "display_name": ["Old QB1", "Confirmed Backup Starter", "QB3"],
            "position": ["QB", "QB", "QB"],
            "status": ["ACT", "ACT", "ACT"],
            "depth_rank": [1, 2, 3],
            "conditional_offense_snap_share": [0.98, 0.20, 0.05],
            "game_day_active_probability": [0.03, 1.0, 1.0],
            "participation_uncertainty": [0.08, 0.12, 0.16],
        }
    )
    usage = pl.DataFrame({"player_id": []}, schema={"player_id": pl.Utf8})

    pool = compile_current_skill_pools(personnel, usage)["ATL"]
    by_id = {player.player_id: player for player in pool.players}

    assert by_id["qb1"].qb_pass_share == 0.0
    assert by_id["qb2"].qb_pass_share > by_id["qb3"].qb_pass_share
    assert by_id["qb2"].qb_pass_share > 0.80
    assert by_id["qb1"].active_probability == 0.03


def test_healthy_qb1_keeps_passing_authority_over_active_backup():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["qb1", "qb2"],
            "pfr_id": [None, None],
            "team_id": ["BUF", "BUF"],
            "display_name": ["Healthy QB1", "Backup"],
            "position": ["QB", "QB"],
            "status": ["ACT", "ACT"],
            "depth_rank": [1, 2],
            "conditional_offense_snap_share": [0.98, 0.05],
            "game_day_active_probability": [0.99, 1.0],
            "participation_uncertainty": [0.08, 0.12],
        }
    )
    usage = pl.DataFrame({"player_id": []}, schema={"player_id": pl.Utf8})

    pool = compile_current_skill_pools(personnel, usage)["BUF"]
    by_id = {player.player_id: player for player in pool.players}

    assert by_id["qb1"].qb_pass_share > 0.99
    assert by_id["qb1"].qb_pass_share > by_id["qb2"].qb_pass_share
