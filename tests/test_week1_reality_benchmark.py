from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load("week1_audit", "scripts/audit_week1_reality_benchmark.py")
truth = _load("week1_truth", "scripts/build_week1_2026_player_truth.py")


def test_empirical_percentile_midrank() -> None:
    assert audit.empirical_percentile([0, 1, 2, 3], 2) == 0.625
    assert audit.empirical_percentile([0, 1, 2, 3], -1) == 0.0
    assert audit.empirical_percentile([0, 1, 2, 3], 4) == 1.0


def test_empirical_crps_degenerate_and_known_case() -> None:
    assert audit.empirical_crps([5, 5, 5], 5) == 0.0
    assert audit.empirical_crps([5, 5, 5], 7) == 2.0
    assert abs(audit.empirical_crps([0, 2], 1) - 0.5) < 1e-12


def test_fanduel_scoring_matches_repo_weights() -> None:
    stats = {
        "passing_yards": 250,
        "passing_tds": 2,
        "interceptions": 1,
        "rushing_yards": 20,
        "rushing_tds": 1,
        "receptions": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "fumbles_lost": 1,
    }
    assert truth.fanduel_points(stats) == 23.0


def test_game_audit_distribution_metrics(tmp_path: Path) -> None:
    sim = tmp_path / "sim"
    sim.mkdir()
    worlds = [
        {"game": "A@B", "away_points": 10, "home_points": 20},
        {"game": "A@B", "away_points": 20, "home_points": 20},
        {"game": "C@D", "away_points": 30, "home_points": 10},
        {"game": "C@D", "away_points": 20, "home_points": 10},
    ]
    with (sim / "football_weirdness_worlds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=worlds[0].keys())
        w.writeheader()
        w.writerows(worlds)
    games = tmp_path / "games.csv"
    truth_rows = [
        {"game": "A@B", "away": "A", "home": "B", "away_points": 15, "home_points": 20, "overtime": "false"},
        {"game": "C@D", "away": "C", "home": "D", "away_points": 25, "home_points": 10, "overtime": "false"},
    ]
    with games.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=truth_rows[0].keys())
        w.writeheader()
        w.writerows(truth_rows)
    rows, summary = audit._game_audit(sim, games)
    assert len(rows) == 2
    assert summary["game_total_mae"] == 0.0
    assert summary["team_points_mae"] == 0.0
    assert summary["mean_within_game_total_sd"] > 0
