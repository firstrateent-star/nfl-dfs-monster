from __future__ import annotations

from pathlib import Path

from monster.ingest.madden import compile_team_madden_inputs, load_team_ratings


def test_official_madden_snapshot_matches_32_team_universe():
    path = Path("data/reference/madden27_team_ratings_2026-07-31.csv")
    ratings = load_team_ratings(path)
    inputs = compile_team_madden_inputs(ratings)
    assert ratings.height == 32
    assert len(inputs) == 32
    # Adapter deliberately uses offense rating in the existing low-authority proxy slot.
    assert inputs["LAR"].team_madden_ovr == 93.0
    assert inputs["MIA"].team_madden_ovr == 74.0
