from __future__ import annotations

import numpy as np
import polars as pl

from monster.feature_compile.participation import compile_snap_priors
from monster.feature_compile.participation_inference import infer_game_day_participation


def test_snap_prior_follows_player_across_team_change():
    snaps = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [15, 16, 17],
            "team": ["A", "A", "A"],
            "position": ["WR", "WR", "WR"],
            "pfr_player_id": ["p1", "p1", "p1"],
            "offense_pct": [80.0, 90.0, 100.0],
            "defense_pct": [0.0, 0.0, 0.0],
            "st_pct": [5.0, 5.0, 5.0],
        }
    )
    row = compile_snap_priors(snaps, recent_games=3).row(0, named=True)
    assert row["prior_team_id"] == "A"
    assert np.isclose(row["offense_snap_share"], 0.90)


def test_active_player_with_recent_snaps_gets_expected_snap_weight():
    frame = pl.DataFrame(
        {
            "status": ["ACT"],
            "position_group": ["WR"],
            "rookie_year": [2022],
            "snap_games_observed": [6],
            "offense_snap_share": [0.80],
            "defense_snap_share": [0.0],
            "special_teams_snap_share": [0.05],
            "snap_share_uncertainty": [0.04],
            "changed_team_since_snap_history": [False],
        }
    )
    row = infer_game_day_participation(frame, season=2026).row(0, named=True)
    # Active-roster status itself is deterministic. Weekly health evidence owns game-day risk.
    assert np.isclose(row["roster_active_probability"], 1.0)
    assert np.isclose(row["game_day_active_probability"], 1.0)
    assert np.isclose(row["projected_offense_snap_share"], 0.80)
    assert row["participation_evidence"] == "recent_snaps"
    assert row["participation_tier"] == "core"


def test_health_availability_remains_the_only_weekly_scratch_probability_for_active_roster():
    frame = pl.DataFrame(
        {
            "status": ["ACT"],
            "position_group": ["QB"],
            "rookie_year": [2022],
            "snap_games_observed": [6],
            "offense_snap_share": [0.95],
            "defense_snap_share": [0.0],
            "special_teams_snap_share": [0.0],
            "snap_share_uncertainty": [0.02],
            "changed_team_since_snap_history": [False],
            "health_availability_probability": [0.72],
            "health_effectiveness_if_active": [0.88],
            "health_uncertainty": [0.20],
        }
    )
    row = infer_game_day_participation(frame, season=2026).row(0, named=True)
    assert np.isclose(row["roster_active_probability"], 1.0)
    assert np.isclose(row["game_day_active_probability"], 0.72)


def test_transfer_and_rookie_uncertainty_are_wider_than_stable_veteran():
    frame = pl.DataFrame(
        {
            "status": ["ACT", "ACT", "ACT"],
            "position_group": ["WR", "WR", "WR"],
            "rookie_year": [2022, 2022, 2026],
            "snap_games_observed": [6, 6, 0],
            "offense_snap_share": [0.75, 0.75, None],
            "defense_snap_share": [0.0, 0.0, None],
            "special_teams_snap_share": [0.05, 0.05, None],
            "snap_share_uncertainty": [0.03, 0.03, None],
            "changed_team_since_snap_history": [False, True, False],
        }
    )
    out = infer_game_day_participation(frame, season=2026)
    stable, transfer, rookie = out.get_column("participation_uncertainty").to_list()
    assert transfer > stable
    assert rookie > stable


def test_practice_squad_and_non_roster_statuses_do_not_dilute_core_units():
    frame = pl.DataFrame(
        {
            "status": ["ACT", "DEV", "RES", "CUT"],
            "position_group": ["DB", "DB", "DB", "DB"],
            "rookie_year": [2022, 2024, 2023, 2023],
            "snap_games_observed": [6, 0, 0, 0],
            "offense_snap_share": [0.0, None, None, None],
            "defense_snap_share": [0.90, None, None, None],
            "special_teams_snap_share": [0.05, None, None, None],
            "snap_share_uncertainty": [0.03, None, None, None],
            "changed_team_since_snap_history": [False, False, False, False],
        }
    )
    out = infer_game_day_participation(frame, season=2026)
    projected = out.get_column("projected_defense_snap_share").to_list()
    assert projected[0] > 0.80
    assert projected[1] < 0.05
    assert projected[2] < 0.01
    assert projected[3] == 0.0


def test_team_unit_priors_are_conserved_toward_eleven_players():
    n = 22
    frame = pl.DataFrame(
        {
            "team_id": ["T"] * n,
            "status": ["ACT"] * n,
            "position_group": ["DB"] * 11 + ["OL"] * 11,
            "rookie_year": [2022] * n,
            "snap_games_observed": [0] * n,
            "offense_snap_share": [None] * n,
            "defense_snap_share": [None] * n,
            "special_teams_snap_share": [None] * n,
            "snap_share_uncertainty": [None] * n,
            "changed_team_since_snap_history": [False] * n,
        }
    )
    out = infer_game_day_participation(frame, season=2026)
    assert 10.8 <= out.get_column("projected_offense_snap_share").sum() <= 11.0
    assert 10.8 <= out.get_column("projected_defense_snap_share").sum() <= 11.0
