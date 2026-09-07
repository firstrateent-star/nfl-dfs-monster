from __future__ import annotations

import polars as pl

from monster.feature_compile.depth import attach_depth_chart, compile_latest_depth_chart
from monster.feature_compile.participation_inference import infer_game_day_participation


def test_latest_depth_chart_uses_best_current_role():
    depth = pl.DataFrame(
        {
            "dt": ["2026-09-01T10:00:00", "2026-09-06T10:00:00", "2026-09-06T10:00:00"],
            "team": ["A", "A", "A"],
            "gsis_id": ["p1", "p1", "p1"],
            "pos_grp": ["WR", "WR", "WR"],
            "pos_abb": ["WR", "WR", "WR"],
            "pos_slot": ["WR", "WR", "SWR"],
            "pos_rank": [2, 2, 1],
        }
    )
    row = compile_latest_depth_chart(depth).row(0, named=True)
    assert row["depth_rank"] == 1
    assert row["depth_slot"] == "SWR"


def test_depth_rank_refines_no_history_role_but_not_observed_snap_prior():
    personnel = pl.DataFrame(
        {
            "team_id": ["A", "A"],
            "gsis_id": ["rook", "vet"],
            "status": ["ACT", "ACT"],
            "position_group": ["WR", "WR"],
            "rookie_year": [2026, 2022],
            "snap_games_observed": [0, 6],
            "offense_snap_share": [None, 0.80],
            "defense_snap_share": [None, 0.0],
            "special_teams_snap_share": [None, 0.05],
            "snap_share_uncertainty": [None, 0.03],
            "changed_team_since_snap_history": [False, False],
        }
    )
    depth = pl.DataFrame(
        {
            "dt": ["2026-09-06T10:00:00", "2026-09-06T10:00:00"],
            "team": ["A", "A"],
            "gsis_id": ["rook", "vet"],
            "pos_grp": ["WR", "WR"],
            "pos_abb": ["WR", "WR"],
            "pos_slot": ["WR", "WR"],
            "pos_rank": [1, 3],
        }
    )
    joined = attach_depth_chart(personnel, depth)
    out = infer_game_day_participation(joined, season=2026)
    rookie, veteran = out.rows(named=True)
    assert rookie["conditional_offense_snap_share"] > 0.36
    assert veteran["conditional_offense_snap_share"] == 0.80
    assert rookie["participation_evidence"] == "depth_prior"
