from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import PBP_COLUMNS
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import NFL_TEAMS, normalize_team_id

STATE_FEATURES = [
    "neutral_pass_rate",
    "neutral_seconds_per_play",
    "drives_per_game",
    "td_drive_rate",
    "fg_drive_rate",
    "turnover_drive_rate",
    "red_zone_td_rate",
    "offensive_epa_per_play",
    "offensive_success_rate",
    "offensive_explosive_rate",
    "defensive_td_drive_rate_allowed",
    "defensive_fg_drive_rate_allowed",
    "defensive_takeaway_drive_rate",
    "defensive_epa_allowed_per_play",
    "defensive_explosive_rate_allowed",
    "defensive_sack_rate",
    "defensive_qb_hit_rate",
]

VARIANTS = {
    "latest": {"shrink": 1.0, "older_weight": 0.0, "trend": 0.0},
    "regress55": {"shrink": 0.55, "older_weight": 0.30, "trend": 0.0},
    "regress70": {"shrink": 0.70, "older_weight": 0.30, "trend": 0.0},
    "regress55_trend50": {"shrink": 0.55, "older_weight": 0.30, "trend": 0.50},
}


def _regular_pbp(season: int) -> pl.DataFrame:
    raw = nfl.load_pbp([season])
    frame = raw.select([c for c in PBP_COLUMNS if c in raw.columns])
    if "season_type" in frame.columns:
        frame = frame.filter(pl.col("season_type") == "REG")
    return frame


def _regular_schedule(season: int) -> pl.DataFrame:
    frame = nfl.load_schedules([season]).filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in frame.columns:
        frame = frame.filter(pl.col("game_type") == "REG")
    return frame


def _canonical_policy(season: int) -> pl.DataFrame:
    compiled = compile_team_policy(_regular_pbp(season))
    return pl.DataFrame({"team_id": list(NFL_TEAMS)}).join(compiled, on="team_id", how="left").sort("team_id")


def _mean(frame: pl.DataFrame, col: str) -> float:
    if col not in frame.columns:
        return 0.0
    values = frame.get_column(col).drop_nulls()
    return float(values.mean()) if len(values) else 0.0


def _build_regressed_policy(
    older: pl.DataFrame,
    latest: pl.DataFrame,
    *,
    shrink: float,
    older_weight: float,
    trend: float,
) -> pl.DataFrame:
    old_rows = {str(r["team_id"]): r for r in older.to_dicts()}
    new_rows = {str(r["team_id"]): r for r in latest.to_dicts()}
    output: list[dict[str, float | str]] = []

    for team in NFL_TEAMS:
        row: dict[str, float | str] = {"team_id": team}
        for feature in STATE_FEATURES:
            latest_mean = _mean(latest, feature)
            older_mean = _mean(older, feature)
            # League environment is anchored to the most recent completed season.
            # A bounded trend candidate asks whether some league-wide movement persists.
            environment = latest_mean + trend * (latest_mean - older_mean)

            latest_value = new_rows.get(team, {}).get(feature)
            older_value = old_rows.get(team, {}).get(feature)
            latest_value = latest_mean if latest_value is None else float(latest_value)
            older_value = older_mean if older_value is None else float(older_value)

            latest_dev = latest_value - latest_mean
            older_dev = older_value - older_mean
            identity_dev = (1.0 - older_weight) * latest_dev + older_weight * older_dev
            row[feature] = float(environment + shrink * identity_dev)
        output.append(row)
    return pl.DataFrame(output).sort("team_id")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_fold(older_season: int, latest_season: int, test_season: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    older = _canonical_policy(older_season)
    latest = _canonical_policy(latest_season)
    policies: dict[str, pl.DataFrame] = {"latest": latest.select(["team_id", *[f for f in STATE_FEATURES if f in latest.columns]])}
    for name, cfg in VARIANTS.items():
        if name == "latest":
            continue
        policies[name] = _build_regressed_policy(older, latest, **cfg)

    schedule = _regular_schedule(test_season)
    records: list[dict] = []
    for idx, game in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(game["away_team"]))
        home = normalize_team_id(str(game["home_team"]))
        paired_seed = seed + idx * 101
        for variant, policy in policies.items():
            states = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
            result = simulate_game(
                GameState(f"{away}@{home}-{variant}", states[away], states[home]),
                worlds=worlds,
                seed=paired_seed,
            )
            records.append({
                "older_season": older_season,
                "latest_season": latest_season,
                "test_season": test_season,
                "variant": variant,
                "game_id": str(game.get("game_id") or f"{away}@{home}"),
                "actual_total": float(game["away_score"] + game["home_score"]),
                "actual_margin": float(game["away_score"] - game["home_score"]),
                "model_total": float((result.away_points + result.home_points).mean()),
                "model_margin": float((result.away_points - result.home_points).mean()),
            })

    frame = pl.DataFrame(records)
    metrics: list[dict] = []
    for variant in VARIANTS:
        sub = frame.filter(pl.col("variant") == variant)
        at, mt = sub["actual_total"].to_numpy(), sub["model_total"].to_numpy()
        am, mm = sub["actual_margin"].to_numpy(), sub["model_margin"].to_numpy()
        metrics.append({
            "older_season": older_season,
            "latest_season": latest_season,
            "test_season": test_season,
            "variant": variant,
            "games": sub.height,
            "total_bias": float((mt - at).mean()),
            "total_mae": float(np.abs(mt - at).mean()),
            "total_corr": _corr(mt, at),
            "margin_mae": float(np.abs(mm - am).mean()),
            "margin_corr": _corr(mm, am),
        })
    return metrics, frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2500)
    p.add_argument("--seed", type=int, default=2026090737)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-regressed-team-state"))
    args = p.parse_args()

    folds = ((2020, 2021, 2022), (2021, 2022, 2023), (2022, 2023, 2024), (2023, 2024, 2025))
    metrics: list[dict] = []
    frames: list[pl.DataFrame] = []
    for i, fold in enumerate(folds):
        m, f = run_fold(*fold, args.worlds, args.seed + i * 10_000_019)
        metrics.extend(m)
        frames.append(f)

    mf = pl.DataFrame(metrics)
    summary = mf.group_by("variant").agg(
        pl.col("total_mae").mean().alias("mean_total_mae"),
        pl.col("total_corr").mean().alias("mean_total_corr"),
        pl.col("margin_mae").mean().alias("mean_margin_mae"),
        pl.col("margin_corr").mean().alias("mean_margin_corr"),
        pl.col("total_bias").mean().alias("mean_total_bias"),
        (pl.col("total_mae") < pl.col("total_mae").filter(pl.col("variant") == "latest").mean()).sum().alias("placeholder"),
    ).drop("placeholder").sort("mean_total_mae")

    baseline = summary.filter(pl.col("variant") == "latest").row(0, named=True)
    candidate_rows: list[dict] = []
    for variant in [v for v in VARIANTS if v != "latest"]:
        row = summary.filter(pl.col("variant") == variant).row(0, named=True)
        folds_v = mf.filter(pl.col("variant") == variant).sort("test_season")
        folds_b = mf.filter(pl.col("variant") == "latest").sort("test_season")
        total_fold_gains = (folds_b["total_mae"] - folds_v["total_mae"]).to_numpy()
        margin_fold_gains = (folds_b["margin_mae"] - folds_v["margin_mae"]).to_numpy()
        candidate_rows.append({
            **row,
            "total_mae_gain_vs_latest": baseline["mean_total_mae"] - row["mean_total_mae"],
            "margin_mae_gain_vs_latest": baseline["mean_margin_mae"] - row["mean_margin_mae"],
            "positive_total_folds": int((total_fold_gains > 0).sum()),
            "positive_margin_folds": int((margin_fold_gains > 0).sum()),
            "positive_both_folds": int(((total_fold_gains > 0) & (margin_fold_gains > 0)).sum()),
        })
    candidates = pl.DataFrame(candidate_rows).sort(
        ["total_mae_gain_vs_latest", "margin_mae_gain_vs_latest"], descending=True
    )
    best = candidates.row(0, named=True)
    robust_pass = bool(
        best["total_mae_gain_vs_latest"] > 0
        and best["margin_mae_gain_vs_latest"] > 0
        and best["mean_total_corr"] >= baseline["mean_total_corr"]
        and best["mean_margin_corr"] >= baseline["mean_margin_corr"]
        and best["positive_both_folds"] >= 3
    )

    manifest = {
        "artifact": "Monster Multi-Season Regressed Team State OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds,
        "paired_common_random_numbers": True,
        "folds": [list(x) for x in folds],
        "variant_definitions": VARIANTS,
        "summary": summary.to_dicts(),
        "candidate_comparison": candidates.to_dicts(),
        "best_candidate": best,
        "robustness_pass": robust_pass,
        "market_blind": True,
        "principle": "League environment and team identity are separate latent states; offseason team identity should regress unless persistent multi-season evidence earns authority.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out / "games.csv")
    mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    candidates.write_csv(args.out / "candidate_comparison.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
