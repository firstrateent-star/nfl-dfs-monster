from __future__ import annotations

import polars as pl

from monster.feature_compile.league_units import compile_league_unit_effects


def test_league_unit_adapter_does_not_double_count_availability():
    snapshot = pl.DataFrame(
        {
            "team_id": ["A", "A"],
            "gsis_id": ["edge", "cb"],
            "position": ["DE", "CB"],
            # These are already expected/conserved snap shares.
            "projected_offense_snap_share": [0.0, 0.0],
            "projected_defense_snap_share": [0.50, 0.50],
            "projected_special_teams_snap_share": [0.0, 0.0],
            "participation_uncertainty": [0.10, 0.10],
            "game_day_active_probability": [0.50, 0.50],
            "observed_pass_rush_signal": [1.0, None],
            "observed_coverage_signal": [None, 1.0],
            "observed_run_defense_signal": [0.5, 0.2],
        }
    )
    row = compile_league_unit_effects(snapshot).row(0, named=True)
    # If active probability were incorrectly multiplied again, trace weight would be 0.5.
    assert row["defense_snap_weight"] == 1.0
    assert row["pass_rush_effect"] > 0.0
    assert row["coverage_effect"] > 0.0


def test_league_unit_compiler_returns_one_row_per_team_and_neutral_missing_offense():
    snapshot = pl.DataFrame(
        {
            "team_id": ["A", "B"],
            "gsis_id": ["a", "b"],
            "position": ["OL", "OL"],
            "projected_offense_snap_share": [0.8, 0.8],
            "projected_defense_snap_share": [0.0, 0.0],
            "projected_special_teams_snap_share": [0.0, 0.0],
            "participation_uncertainty": [0.1, 0.1],
        }
    )
    out = compile_league_unit_effects(snapshot)
    assert out.height == 2
    assert out.get_column("team_id").to_list() == ["A", "B"]
    assert out.get_column("pass_protection_effect").to_list() == [0.0, 0.0]
    assert out.get_column("run_block_effect").to_list() == [0.0, 0.0]
