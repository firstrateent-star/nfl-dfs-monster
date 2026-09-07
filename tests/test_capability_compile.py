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
