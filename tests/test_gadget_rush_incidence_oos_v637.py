from __future__ import annotations

import polars as pl

from monster.feature_compile.gadget_rush_incidence_v637 import (
    normalize_weekly_stats,
    summarize_incidence,
    transition_table,
)


def _raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2, 1, 1, 2, 2],
            "season_type": ["REG"] * 8,
            "player_id": ["wr", "rb", "wr", "rb", "wr", "rb", "wr", "rb"],
            "position": ["WR", "RB", "WR", "RB", "WR", "RB", "WR", "RB"],
            "recent_team": ["A"] * 8,
            "carries": [1, 19, 0, 20, 0, 20, 1, 19],
        }
    )


def test_team_denominator_is_built_before_position_filter() -> None:
    stats = normalize_weekly_stats(_raw())
    wr = stats.filter(pl.col("position") == "WR").sort(["season", "week"])
    assert wr["team_carries"].to_list() == [20.0, 20.0, 20.0, 20.0]
    assert wr["rush_entry"].to_list() == [1, 0, 0, 1]


def test_incidence_summary_and_transition_preserve_game_role_state() -> None:
    stats = normalize_weekly_stats(_raw())
    summary = summarize_incidence(stats)
    wr = (
        summary["position_summary"]
        .filter(pl.col("position") == "WR")
        .row(0, named=True)
    )
    assert wr["player_games"] == 4
    assert wr["rush_entry_games"] == 2
    assert abs(wr["entry_rate"] - 0.5) < 1e-12

    transitions, grouped = transition_table(summary["player_season"])
    assert transitions.height == 1
    row = transitions.row(0, named=True)
    assert row["prior_rush_games"] == 1
    assert row["next_rush_games"] == 1
    assert row["prior_carry_bin"] == "1-3"

    grouped_row = grouped.row(0, named=True)
    assert abs(grouped_row["weighted_next_entry_rate"] - 0.5) < 1e-12
