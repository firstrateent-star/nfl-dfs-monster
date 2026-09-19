from __future__ import annotations

import polars as pl

from monster.reality.qb_rush_audit import (
    classify_qb_rush_rows,
    summarize_qb_rush_family_worlds,
)


def _snaps() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game": ["A@B"] * 6,
            "world": [0] * 6,
            "play_type": ["pass", "run", "run", "pass", "pass", "run"],
            "pass_result": ["complete", None, None, "scramble", "scramble", None],
            "passer_id": ["qb", None, None, "qb", "qb", None],
            "rusher_id": [None, "qb", "rb", "qb", "qb", "qb"],
            "run_geometry": [None, "left_edge", "interior", None, None, "qb_sneak"],
            "pressured": [False, False, False, True, False, False],
            "yards": [8.0, 7.0, 4.0, 6.0, 11.0, 1.0],
        }
    )


def test_simulated_qb_rush_family_classification_separates_paths() -> None:
    classified = classify_qb_rush_rows(_snaps())
    assert classified.height == 4
    counts = {
        row["qb_rush_family"]: row["len"]
        for row in classified.group_by("qb_rush_family").agg(pl.len()).to_dicts()
    }
    assert counts == {
        "designed_non_sneak": 1,
        "sneak": 1,
        "pressure_scramble": 1,
        "coverage_scramble": 1,
    }


def test_simulated_qb_rush_family_summary_conserves_attempts_and_yards() -> None:
    worlds = summarize_qb_rush_family_worlds(classify_qb_rush_rows(_snaps()))
    assert worlds.height == 1
    row = worlds.to_dicts()[0]
    assert row["designed_non_sneak_attempts"] == 1
    assert row["sneak_attempts"] == 1
    assert row["pressure_scramble_attempts"] == 1
    assert row["coverage_scramble_attempts"] == 1
    assert row["competitive_qb_rush_attempts"] == 4
    assert row["competitive_qb_rush_yards"] == 25.0
    assert row["scramble_attempts"] == 2
