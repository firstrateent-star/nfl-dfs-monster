from __future__ import annotations

import polars as pl

from monster.ingest.madden_official import attach_all_madden_attributes
from monster.ingest.madden_schema import legacy_madden_adapter_view, normalize_madden_team
from monster.tabular import csv_safe_frame


def test_official_madden_matches_team_names_transfers_and_refuses_ambiguity():
    personnel = pl.DataFrame(
        {
            "display_name": ["D.J. Test Jr.", "Transfer Runner", "Shared Name"],
            "full_name": ["DJ Test", "Transfer Runner", "Shared Name"],
            "team_id": ["BUF", "PHI", "BUF"],
            "position": ["WR", "RB", "WR"],
            "position_group": ["WR", "RB", "WR"],
        }
    )
    ratings = pl.DataFrame(
        {
            "madden_player_id": [1, 2, 3, 4],
            "madden_player_name": ["DJ Test", "Transfer Runner", "Shared Name", "Shared Name"],
            "madden_match_name_key": ["djtest", "transferrunner", "sharedname", "sharedname"],
            "madden_team": ["Buffalo Bills", "Dallas Cowboys", "New York Jets", "New York Giants"],
            "madden_position": ["WR", "HB", "WR", "WR"],
            "madden_speed": [91.0, 88.0, 80.0, 81.0],
            "madden_ability_1": [["Deep Threat"], ["Workhorse"], ["A"], ["B"]],
        }
    )

    attached = attach_all_madden_attributes(personnel, ratings)
    assert attached.height == personnel.height
    rows = attached.to_dicts()

    assert rows[0]["madden_player_id"] == 1
    assert rows[0]["madden_official_match_type"] == "team_name"
    assert rows[0]["madden_speed"] == 91.0
    assert rows[0]["madden_ability_1"] == ["Deep Threat"]

    assert rows[1]["madden_player_id"] == 2
    assert rows[1]["madden_official_match_type"] == "name_position"

    assert rows[2]["madden_player_id"] is None
    assert rows[2]["madden_official_match_type"] is None


def test_madden_team_normalization_handles_ea_franchise_and_nickname_forms():
    assert normalize_madden_team("Buffalo Bills") == "BUF"
    assert normalize_madden_team("Bills") == "BUF"
    assert normalize_madden_team("JAX") == "JAC"
    assert normalize_madden_team("WSH") == "WAS"


def test_legacy_adapter_view_exposes_old_contract_without_removing_canonical_fields():
    ratings = pl.DataFrame(
        {
            "madden_player_name": ["Example Player"],
            "madden_position": ["WR"],
            "madden_team": ["Buffalo Bills"],
            "madden_speed": [92.0],
            "madden_catching": [87.0],
            "madden_short_route_running": [84.0],
        }
    )
    legacy = legacy_madden_adapter_view(ratings)
    assert legacy.get_column("full_name").item() == "Example Player"
    assert legacy.get_column("position").item() == "WR"
    assert legacy.get_column("team_name").item() == "Buffalo Bills"
    assert legacy.get_column("speed_rating").item() == 92.0
    assert legacy.get_column("madden_speed").item() == 92.0


def test_csv_safe_frame_serializes_nested_columns_without_mutating_source():
    frame = pl.DataFrame({"player": ["A"], "ability": [["X", "Y"]]})
    safe = csv_safe_frame(frame)
    assert frame.schema["ability"].base_type() == pl.List
    assert safe.schema["ability"] == pl.Utf8
    assert "X" in safe.get_column("ability").item()
