from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from validate_oos_regressed_team_state import (
    STATE_FEATURES,
    _canonical_policy,
    _corr,
    _mean,
    _regular_schedule,
)
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import NFL_TEAMS, normalize_team_id

BEHAVIOR = {
    "neutral_pass_rate",
    "neutral_seconds_per_play",
    "drives_per_game",
}
CONVERSION = {
    "td_drive_rate",
    "fg_drive_rate",
    "turnover_drive_rate",
    "red_zone_td_rate",
    "defensive_td_drive_rate_allowed",
    "defensive_fg_drive_rate_allowed",
    "defensive_takeaway_drive_rate",
}
OFFENSE_EFF = {
    "offensive_epa_per_play",
    "offensive_success_rate",
    "offensive_explosive_rate",
}
DEFENSE_EFF = {
    "defensive_epa_allowed_per_play",
    "defensive_explosive_rate_allowed",
    "defensive_sack_rate",
    "defensive_qb_hit_rate",
}

VARIANTS: dict[str, set[str]] = {
    "latest": set(),
    "behavior55": BEHAVIOR,
    "conversion55": CONVERSION,
    "offense55": OFFENSE_EFF,
    "defense55": DEFENSE_EFF,
    "efficiency55": OFFENSE_EFF | DEFENSE_EFF,
    "conversion_efficiency55": CONVERSION | OFFENSE_EFF | DEFENSE_EFF,
    "all55": set(STATE_FEATURES),
}


def _component_policy(
    older: pl.DataFrame,
    latest: pl.DataFrame,
    shrink_features: set[str],
    *,
    shrink: float = 0.55,
    older_weight: float = 0.30,
) -> pl.DataFrame:
    old_rows = {str(r["team_id"]): r for r in older.to_dicts()}
    new_rows = {str(r["team_id"]): r for r in latest.to_dicts()}
    output: list[dict[str, float | str]] = []
    for team in NFL_TEAMS:
        row: dict[str, float | str] = {"team_id": team}
        for feature in STATE_FEATURES:
            lm = _mean(latest, feature)
            om = _mean(older, feature)
            lv = new_rows.get(team, {}).get(feature)
            ov = old_rows.get(team, {}).get(feature)
            lv = lm if lv is None else float(lv)
            ov = om if ov is None else float(ov)
            if feature in shrink_features:
                latest_dev = lv - lm
                older_dev = ov - om
                identity_dev = (1.0 - older_weight) * latest_dev + older_weight * older_dev
                row[feature] = float(lm + shrink * identity_dev)
            else:
                row[feature] = float(lv)
        output.append(row)
    return pl.DataFrame(output).sort("team_id")


def run_fold(older_season: int, latest_season: int, test_season: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    older = _canonical_policy(older_season)
    latest = _canonical_policy(latest_season)
    policies = {
        name: (
            latest.select(["team_id", *[f for f in STATE_FEATURES if f in latest.columns]])
            if name == "latest"
            else _component_policy(older, latest, features)
        )
        for name, features in VARIANTS.items()
    }
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
    p.add_argument("--worlds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026090743)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-component-regression"))
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
    ).sort("mean_total_mae")
    baseline = summary.filter(pl.col("variant") == "latest").row(0, named=True)

    comparisons: list[dict] = []
    for variant in [v for v in VARIANTS if v != "latest"]:
        row = summary.filter(pl.col("variant") == variant).row(0, named=True)
        fv = mf.filter(pl.col("variant") == variant).sort("test_season")
        fb = mf.filter(pl.col("variant") == "latest").sort("test_season")
        tg = (fb["total_mae"] - fv["total_mae"]).to_numpy()
        mg = (fb["margin_mae"] - fv["margin_mae"]).to_numpy()
        comparisons.append({
            **row,
            "total_mae_gain": baseline["mean_total_mae"] - row["mean_total_mae"],
            "margin_mae_gain": baseline["mean_margin_mae"] - row["mean_margin_mae"],
            "total_corr_gain": row["mean_total_corr"] - baseline["mean_total_corr"],
            "margin_corr_gain": row["mean_margin_corr"] - baseline["mean_margin_corr"],
            "positive_total_folds": int((tg > 0).sum()),
            "positive_margin_folds": int((mg > 0).sum()),
            "positive_both_folds": int(((tg > 0) & (mg > 0)).sum()),
        })
    cf = pl.DataFrame(comparisons).sort(["positive_both_folds", "total_mae_gain", "margin_mae_gain"], descending=True)
    best = cf.row(0, named=True)
    passed = bool(
        best["total_mae_gain"] > 0
        and best["margin_mae_gain"] > 0
        and best["total_corr_gain"] >= 0
        and best["margin_corr_gain"] >= 0
        and best["positive_both_folds"] >= 3
    )
    manifest = {
        "artifact": "Monster Component-wise Offseason Regression OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds,
        "paired_common_random_numbers": True,
        "folds": [list(x) for x in folds],
        "feature_groups": {
            "behavior": sorted(BEHAVIOR),
            "conversion": sorted(CONVERSION),
            "offense_efficiency": sorted(OFFENSE_EFF),
            "defense_efficiency": sorted(DEFENSE_EFF),
        },
        "summary": summary.to_dicts(),
        "candidate_comparison": cf.to_dicts(),
        "best_candidate": best,
        "robustness_pass": passed,
        "market_blind": True,
        "principle": "Different football mechanisms may persist across an offseason at different rates; regression authority must be mechanism-specific and OOS-earned.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out / "games.csv")
    mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    cf.write_csv(args.out / "candidate_comparison.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
