from __future__ import annotations

import polars as pl
import pytest

from scripts.audit_v723_premise_robustness import build_premise_robustness
from scripts.benchmark_v723_week2_reality import build_reality_benchmark


def test_premise_robustness_separates_offensive_regimes() -> None:
    regimes = (
        "collapse",
        "collapse",
        "fragile",
        "fragile",
        "normal",
        "normal",
        "surge",
        "surge",
    )
    points = (4.0, 6.0, 9.0, 11.0, 14.0, 16.0, 34.0, 36.0)
    player_world = pl.DataFrame(
        {
            "game": ["AAA@BBB"] * 8,
            "world": list(range(8)),
            "player_id": ["p1"] * 8,
            "player": ["Player One"] * 8,
            "position": ["WR"] * 8,
            "team": ["AAA"] * 8,
            "fanduel_points": points,
            "pass_attempts": [0.0] * 8,
            "targets": [1.0] * 8,
            "rush_attempts": [0.0] * 8,
        }
    )
    premise_rows: list[dict[str, object]] = []
    for world, regime in enumerate(regimes):
        premise_rows.extend(
            [
                {
                    "game": "AAA@BBB",
                    "world": world,
                    "team": "AAA",
                    "regime": regime,
                    "offensive_cohesion": float(world - 4) / 4.0,
                    "defensive_regime": "normal",
                    "defensive_cohesion": 0.0,
                },
                {
                    "game": "AAA@BBB",
                    "world": world,
                    "team": "BBB",
                    "regime": "normal",
                    "offensive_cohesion": 0.0,
                    "defensive_regime": regime,
                    "defensive_cohesion": float(world - 4) / 4.0,
                },
            ]
        )

    report = build_premise_robustness(
        player_world,
        pl.DataFrame(premise_rows),
        minimum_regime_worlds=2,
    )
    row = report.row(0, named=True)
    assert row["collapse_fd_mean"] == pytest.approx(5.0)
    assert row["normal_fd_mean"] == pytest.approx(15.0)
    assert row["surge_fd_mean"] == pytest.approx(35.0)
    assert row["worst_regime_fd_mean"] == pytest.approx(5.0)
    assert row["regime_fd_mean_span"] == pytest.approx(30.0)
    assert row["collapse_retention_vs_normal"] == pytest.approx(1.0 / 3.0)


def test_week2_reality_benchmark_measures_score_and_interval_error() -> None:
    projected = pl.DataFrame(
        {
            "game": ["AAA@BBB", "CCC@DDD"],
            "away": ["AAA", "CCC"],
            "home": ["BBB", "DDD"],
            "projected_winner": ["AAA", "CCC"],
            "away_points_mean": [24.0, 27.0],
            "home_points_mean": [20.0, 21.0],
            "away_points_p10": [14.0, 17.0],
            "away_points_p90": [34.0, 37.0],
            "home_points_p10": [10.0, 11.0],
            "home_points_p90": [30.0, 31.0],
            "total_mean": [44.0, 48.0],
            "margin_mean": [4.0, 6.0],
        }
    )
    actual = pl.DataFrame(
        {
            "game": ["AAA@BBB", "CCC@DDD"],
            "away": ["AAA", "CCC"],
            "home": ["BBB", "DDD"],
            "actual_away_points": [28, 10],
            "actual_home_points": [17, 24],
        }
    )

    _, summary = build_reality_benchmark(projected, actual)
    assert summary["games"] == 2
    assert summary["winner_accuracy"] == pytest.approx(0.5)
    assert summary["team_scores_below_p10"] == 1
    assert summary["team_scores_above_p90"] == 0
    assert summary["projected_game_total_mean"] == pytest.approx(46.0)
    assert summary["actual_game_total_mean"] == pytest.approx(39.5)
    assert summary["game_total_bias"] == pytest.approx(6.5)
