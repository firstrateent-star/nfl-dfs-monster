from __future__ import annotations

import polars as pl

from monster.feature_compile.situation import compile_situational_pass_context


def test_context_compiler_builds_all_down_distance_cells() -> None:
    rows = []
    for down in range(1, 5):
        for ydstogo in (2, 5, 10):
            rows.extend(
                [
                    {
                        "posteam": "A",
                        "play_type": "pass",
                        "qb_dropback": 1,
                        "down": down,
                        "ydstogo": ydstogo,
                        "score_differential": 0,
                        "game_seconds_remaining": 2400,
                    },
                    {
                        "posteam": "A",
                        "play_type": "run",
                        "qb_dropback": 0,
                        "down": down,
                        "ydstogo": ydstogo,
                        "score_differential": 0,
                        "game_seconds_remaining": 2400,
                    },
                ]
            )
    context = compile_situational_pass_context(pl.DataFrame(rows))
    assert context.height == 12
    assert set(context.get_column("distance_bucket")) == {"short", "medium", "long"}
    assert context.get_column("samples").min() == 2
    assert context.get_column("pass_rate").min() == 0.5
    assert context.get_column("pass_rate").max() == 0.5


def test_scramble_is_pass_family_not_designed_run() -> None:
    frame = pl.DataFrame(
        [
            {
                "posteam": "A",
                "play_type": "run",
                "qb_dropback": 1,
                "down": 1,
                "ydstogo": 10,
                "score_differential": 0,
                "game_seconds_remaining": 2400,
            },
            {
                "posteam": "A",
                "play_type": "run",
                "qb_dropback": 0,
                "down": 1,
                "ydstogo": 10,
                "score_differential": 0,
                "game_seconds_remaining": 2400,
            },
        ]
    )
    context = compile_situational_pass_context(frame)
    first_long = context.filter(
        (pl.col("down") == 1) & (pl.col("distance_bucket") == "long")
    )
    assert first_long.get_column("samples").item() == 2
    assert first_long.get_column("pass_rate").item() == 0.5


def test_context_normalizes_nullable_float_down_from_provider() -> None:
    frame = pl.DataFrame(
        {
            "posteam": ["A", "A", "A"],
            "play_type": ["pass", "run", "pass"],
            "qb_dropback": [1.0, 0.0, 1.0],
            "down": pl.Series([1.0, 1.0, None], dtype=pl.Float64),
            "ydstogo": [10.0, 10.0, 7.0],
            "score_differential": [0.0, 0.0, 0.0],
            "game_seconds_remaining": [2400.0, 2400.0, 2400.0],
        }
    )
    context = compile_situational_pass_context(frame)
    first_long = context.filter(
        (pl.col("down") == 1) & (pl.col("distance_bucket") == "long")
    )
    assert context.schema["down"] == pl.Int64
    assert context.height == 12
    assert first_long.get_column("samples").item() == 2
    assert first_long.get_column("pass_rate").item() == 0.5


def test_context_uses_neutral_game_states_only() -> None:
    frame = pl.DataFrame(
        [
            {
                "posteam": "A",
                "play_type": "run",
                "qb_dropback": 0,
                "down": 1,
                "ydstogo": 10,
                "score_differential": 0,
                "game_seconds_remaining": 2400,
            },
            {
                "posteam": "A",
                "play_type": "pass",
                "qb_dropback": 1,
                "down": 1,
                "ydstogo": 10,
                "score_differential": -21,
                "game_seconds_remaining": 2400,
            },
        ]
    )
    context = compile_situational_pass_context(frame)
    first_long = context.filter(
        (pl.col("down") == 1) & (pl.col("distance_bucket") == "long")
    )
    assert first_long.get_column("pass_rate").item() == 0.0
