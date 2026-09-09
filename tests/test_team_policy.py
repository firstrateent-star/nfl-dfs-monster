from __future__ import annotations

import polars as pl

from monster.feature_compile.team import compile_team_policy


def _row(
    *,
    posteam: str,
    defteam: str,
    play_type: str,
    qb_dropback: int,
    down: int,
    seconds: int,
    drive: int,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_A_B",
        "posteam": posteam,
        "defteam": defteam,
        "play_type": play_type,
        "down": down,
        "yardline_100": 50.0,
        "game_seconds_remaining": seconds,
        "score_differential": 0.0,
        "epa": 0.0,
        "success": 0.0,
        "touchdown": 0,
        "pass_touchdown": 0,
        "rush_touchdown": 0,
        "interception": 0,
        "fumble_lost": 0,
        "sack": 0,
        "qb_hit": 0,
        "qb_dropback": qb_dropback,
        "field_goal_attempt": 0,
        "field_goal_result": "none",
        "fixed_drive": drive,
        "yards_gained": 4.0,
    }


def test_team_neutral_pass_rate_counts_scramble_as_dropback_family() -> None:
    pbp = pl.DataFrame(
        [
            _row(
                posteam="A",
                defteam="B",
                play_type="run",
                qb_dropback=1,
                down=1,
                seconds=2400,
                drive=1,
            ),
            _row(
                posteam="A",
                defteam="B",
                play_type="run",
                qb_dropback=0,
                down=2,
                seconds=2370,
                drive=1,
            ),
            _row(
                posteam="B",
                defteam="A",
                play_type="pass",
                qb_dropback=1,
                down=1,
                seconds=2300,
                drive=2,
            ),
            _row(
                posteam="B",
                defteam="A",
                play_type="run",
                qb_dropback=0,
                down=2,
                seconds=2270,
                drive=2,
            ),
        ]
    )

    policy = compile_team_policy(pbp)
    team_a = policy.filter(pl.col("team_id") == "A")

    assert team_a.get_column("neutral_plays").item() == 2
    assert team_a.get_column("neutral_pass_rate").item() == 0.5
    assert team_a.get_column("early_down_pass_rate").item() == 0.5
