from __future__ import annotations

import polars as pl

from monster.feature_compile.capability import (
    attach_capability_evidence,
    compile_combine_features,
    compile_defender_history,
)


def test_combine_features_preserve_raw_measurements_and_speed_score():
    combine = pl.DataFrame(
        {
            "pfr_id": ["p1"],
            "wt": [220.0],
            "forty": [4.40],
            "bench": [20.0],
            "vertical": [38.0],
            "broad_jump": [125.0],
            "cone": [7.0],
            "shuttle": [4.2],
        }
    )
    row = compile_combine_features(combine).row(0, named=True)
    assert row["forty"] == 4.40
    assert row["combine_speed_score"] > 100.0


def test_defender_history_keeps_pressure_and_coverage_channels_separate():
    defense = pl.DataFrame(
        {
            "pfr_player_id": ["d1", "d1"],
            "def_times_hurried": [2.0, 3.0],
            "def_times_hitqb": [1.0, 2.0],
            "def_pass_defended": [1.0, 2.0],
            "def_times_pressured_pct": [10.0, 20.0],
        }
    )
    row = compile_defender_history(defense).row(0, named=True)
    assert row["hist_def_times_hurried"] == 5.0
    assert row["hist_def_times_hitqb"] == 3.0
    assert row["hist_def_pass_defended"] == 3.0
    assert row["hist_def_times_pressured_pct"] == 15.0


def test_capability_attachment_is_neutral_when_evidence_is_missing():
    personnel = pl.DataFrame({"pfr_id": ["p1"], "team_id": ["A"]})
    out = attach_capability_evidence(personnel, pl.DataFrame(), pl.DataFrame())
    assert out.get_column("defense_games_observed").to_list() == [0]
    assert out.get_column("observed_special_teams_signal").to_list() == [None]


def test_kicking_and_return_evidence_create_separate_special_teams_signals():
    personnel = pl.DataFrame(
        {
            "pfr_id": ["k1", "k2", "r1", "r2"],
            "gsis_id": ["k1", "k2", "r1", "r2"],
            "team_id": ["A", "B", "A", "B"],
            "position_group": ["SPEC", "SPEC", "WR", "WR"],
            "position": ["K", "K", "WR", "WR"],
        }
    )
    stats = pl.DataFrame(
        {
            "player_id": ["k1", "k2", "r1", "r2"],
            "fg_made": [30.0, 20.0, 0.0, 0.0],
            "fg_att": [32.0, 30.0, 0.0, 0.0],
            "fg_made_40_49": [8.0, 5.0, 0.0, 0.0],
            "fg_made_50_59": [6.0, 2.0, 0.0, 0.0],
            "fg_made_60_": [1.0, 0.0, 0.0, 0.0],
            "pat_made": [40.0, 40.0, 0.0, 0.0],
            "pat_att": [41.0, 41.0, 0.0, 0.0],
            "punt_returns": [0.0, 0.0, 15.0, 15.0],
            "punt_return_yards": [0.0, 0.0, 180.0, 90.0],
            "kickoff_returns": [0.0, 0.0, 8.0, 8.0],
            "kickoff_return_yards": [0.0, 0.0, 220.0, 150.0],
        }
    )
    out = attach_capability_evidence(personnel, pl.DataFrame(), pl.DataFrame(), stats)
    rows = {r["gsis_id"]: r for r in out.to_dicts()}
    assert rows["k1"]["observed_kicking_signal"] > rows["k2"]["observed_kicking_signal"]
    assert rows["r1"]["observed_return_signal"] > rows["r2"]["observed_return_signal"]
    assert rows["k1"]["observed_special_teams_signal"] is not None
    assert rows["r1"]["observed_special_teams_signal"] is not None
