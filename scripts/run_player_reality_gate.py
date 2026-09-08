from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/player-reality-audit"))
    args = parser.parse_args()

    p = pl.read_csv(args.players)
    t = pl.read_csv(args.teams)
    args.out.mkdir(parents=True, exist_ok=True)

    starters = p.filter((pl.col("position") == "QB") & (pl.col("pass_attempts_mean") >= 15.0))
    qb_attempts = starters.get_column("pass_attempts_mean").to_numpy()
    qb_yards = starters.get_column("passing_yards_mean").to_numpy()
    qb_tds = starters.get_column("passing_tds_mean").to_numpy()

    rb = p.filter(pl.col("position") == "RB")
    wrte = p.filter(pl.col("position").is_in(["WR", "TE"]))

    team_plays = t.get_column("plays_mean").to_numpy()
    team_pass = t.get_column("pass_attempts_mean").to_numpy()
    team_rush = t.get_column("rush_attempts_mean").to_numpy()
    team_targets = t.get_column("targets_mean").to_numpy()
    team_ypp = t.get_column("yards_per_play_mean").to_numpy()
    target1 = t.get_column("target_rank1_share_world_mean").to_numpy()
    target2 = t.get_column("target_rank2_share_world_mean").to_numpy()
    target3 = t.get_column("target_rank3_share_world_mean").to_numpy()
    rush1 = t.get_column("rush_rank1_share_world_mean").to_numpy()
    rush2 = t.get_column("rush_rank2_share_world_mean").to_numpy()
    rush3 = t.get_column("rush_rank3_share_world_mean").to_numpy()
    target_earners = t.get_column("target_earners_per_world_mean").to_numpy()
    core_rushers = t.get_column("core_rushers_per_world_mean").to_numpy()

    finite_cols = [
        "fd_points_before_turnover_penalties_mean",
        "pass_attempts_mean", "passing_yards_mean", "passing_tds_mean",
        "targets_mean", "receptions_mean", "receiving_yards_mean", "receiving_tds_mean",
        "rush_attempts_mean", "rushing_yards_mean", "rushing_tds_mean",
    ]
    finite_ok = True
    for col in finite_cols:
        if col in p.columns:
            finite_ok &= bool(np.isfinite(p.get_column(col).to_numpy()).all())

    # These are broad structural sanity gates, deliberately wider than the historical centers.
    # Their purpose is to catch impossible allocation/conservation states, not tune projections.
    checks = {
        "finite_player_outputs": finite_ok,
        "team_plays_sane": bool(np.all((team_plays >= 48.0) & (team_plays <= 78.0))),
        "team_pass_attempts_sane": bool(np.all((team_pass >= 20.0) & (team_pass <= 48.0))),
        "team_rush_attempts_sane": bool(np.all((team_rush >= 16.0) & (team_rush <= 42.0))),
        "team_targets_le_pass_attempts": bool(np.all(team_targets <= team_pass + 1e-6)),
        "team_yards_per_play_sane": bool(np.all((team_ypp >= 3.5) & (team_ypp <= 7.5))),
        "target_rank1_sane": bool(np.all((target1 >= 0.20) & (target1 <= 0.42))),
        "target_rank2_sane": bool(np.all((target2 >= 0.12) & (target2 <= 0.32))),
        "target_rank3_sane": bool(np.all((target3 >= 0.08) & (target3 <= 0.25))),
        "rush_rank1_sane": bool(np.all((rush1 >= 0.35) & (rush1 <= 0.78))),
        "rush_rank2_sane": bool(np.all((rush2 >= 0.12) & (rush2 <= 0.42))),
        "rush_rank3_sane": bool(np.all((rush3 >= 0.03) & (rush3 <= 0.25))),
        "target_breadth_sane": bool(np.all((target_earners >= 4.0) & (target_earners <= 10.0))),
        "core_rusher_breadth_sane": bool(np.all((core_rushers >= 1.5) & (core_rushers <= 4.0))),
        "starter_qb_count_sane": bool(20 <= starters.height <= 24),
        "starter_qb_attempts_sane": bool(np.all((qb_attempts >= 20.0) & (qb_attempts <= 45.0))),
        "starter_qb_passing_yards_sane": bool(np.all((qb_yards >= 120.0) & (qb_yards <= 360.0))),
        "starter_qb_passing_tds_sane": bool(np.all((qb_tds >= 0.4) & (qb_tds <= 3.2))),
        "rb_nonnegative": bool((rb.select(pl.col("rush_attempts_mean").min()).item() or 0.0) >= 0.0),
        "wrte_nonnegative": bool((wrte.select(pl.col("targets_mean").min()).item() or 0.0) >= 0.0),
    }

    metrics = {
        "player_rows": p.height,
        "team_rows": t.height,
        "starter_qb_count": starters.height,
        "team_plays": {"min": float(team_plays.min()), "median": float(np.median(team_plays)), "max": float(team_plays.max())},
        "team_pass_attempts": {"min": float(team_pass.min()), "median": float(np.median(team_pass)), "max": float(team_pass.max())},
        "team_rush_attempts": {"min": float(team_rush.min()), "median": float(np.median(team_rush)), "max": float(team_rush.max())},
        "team_targets": {"min": float(team_targets.min()), "median": float(np.median(team_targets)), "max": float(team_targets.max())},
        "yards_per_play": {"min": float(team_ypp.min()), "median": float(np.median(team_ypp)), "max": float(team_ypp.max())},
        "target_rank_shares_mean": [float(target1.mean()), float(target2.mean()), float(target3.mean())],
        "rush_rank_shares_mean": [float(rush1.mean()), float(rush2.mean()), float(rush3.mean())],
        "target_earners_mean": float(target_earners.mean()),
        "core_rushers_mean": float(core_rushers.mean()),
        "starter_qb_pass_attempts": {"median": float(np.median(qb_attempts)), "p90": _q(qb_attempts, .90)},
        "starter_qb_passing_yards": {"median": float(np.median(qb_yards)), "p90": _q(qb_yards, .90)},
        "starter_qb_passing_tds": {"median": float(np.median(qb_tds)), "p90": _q(qb_tds, .90)},
    }
    gate_pass = bool(all(checks.values()))
    manifest = {
        "artifact": "Monster Player Reality Broad Structural Gate",
        "gate_pass": gate_pass,
        "checks": checks,
        "metrics": metrics,
        "principle": "Broad gates reject impossible or structurally implausible player allocation without tuning football toward DFS or market outputs.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    t.write_csv(args.out / "team_opportunity_map.csv")
    starters.sort("pass_attempts_mean", descending=True).write_csv(args.out / "starter_qbs.csv")
    print(json.dumps(manifest, indent=2))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
