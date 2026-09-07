from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.ingest.league import build_league_personnel_snapshot
from monster.ingest.nflverse import load_league_personnel_inputs
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import NFL_TEAMS, normalize_team_id
from validate_oos_offensive_personnel import _week1, _optional_depth
from validate_oos_regressed_team_state import STATE_FEATURES, _canonical_policy, _corr, _mean, _regular_schedule

BEHAVIOR = {"neutral_pass_rate", "neutral_seconds_per_play", "drives_per_game"}
VARIANTS = ("latest", "fixed55", "continuity_all", "continuity_nonbehavior")


def _team_continuity(train: int, test: int) -> tuple[dict[str, float], bool]:
    inputs = load_league_personnel_inputs([train], test, Path(".cache/monster"))
    snapshot = build_league_personnel_snapshot(
        _week1(inputs["current_rosters"]),
        inputs["players"],
        inputs["historical_snap_counts"],
        None,
        recent_games=6,
    )
    snapshot, depth_used = _optional_depth(snapshot, inputs["depth_charts"])
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    core = pos.is_in(["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"])
    snap = pl.max_horizontal(
        pl.col("offense_snap_share").fill_null(0.0) if "offense_snap_share" in snapshot.columns else pl.lit(0.0),
        pl.col("defense_snap_share").fill_null(0.0) if "defense_snap_share" in snapshot.columns else pl.lit(0.0),
    ).clip(0.0, 1.0)
    depth_w = pl.when(pl.col("depth_rank").is_not_null()).then(
        (1.0 / pl.col("depth_rank").cast(pl.Float64)).clip(0.20, 1.0)
    ).otherwise(0.30)
    w = pl.max_horizontal(snap, 0.30 * depth_w)
    retained = (
        core
        & pl.col("prior_team_id").is_not_null()
        & (pl.col("prior_team_id") == pl.col("team_id"))
        & (pl.col("snap_games_observed").fill_null(0) > 0)
    )
    rows = snapshot.with_columns(w.alias("_w")).group_by("team_id").agg(
        pl.when(core).then(pl.col("_w")).otherwise(0.0).sum().alias("total_w"),
        pl.when(retained).then(pl.col("_w")).otherwise(0.0).sum().alias("retained_w"),
    ).to_dicts()
    out = {t: 0.50 for t in NFL_TEAMS}
    for r in rows:
        total = max(float(r["total_w"] or 0.0), 1e-9)
        out[str(r["team_id"])] = float(np.clip(float(r["retained_w"] or 0.0) / total, 0.0, 1.0))
    return out, depth_used


def _build_policy(older: pl.DataFrame, latest: pl.DataFrame, continuity: dict[str, float], variant: str) -> pl.DataFrame:
    old_rows = {str(r["team_id"]): r for r in older.to_dicts()}
    new_rows = {str(r["team_id"]): r for r in latest.to_dicts()}
    output: list[dict[str, float | str]] = []
    for team in NFL_TEAMS:
        row: dict[str, float | str] = {"team_id": team}
        c = float(continuity.get(team, 0.50))
        dynamic = 0.30 + 0.60 * c  # 0.30..0.90 memory authority
        for feature in STATE_FEATURES:
            lm = _mean(latest, feature); om = _mean(older, feature)
            lv = new_rows.get(team, {}).get(feature); ov = old_rows.get(team, {}).get(feature)
            lv = lm if lv is None else float(lv); ov = om if ov is None else float(ov)
            identity = 0.75 * (lv - lm) + 0.25 * (ov - om)
            if variant == "latest":
                shrink = 1.0; identity = lv - lm
            elif variant == "fixed55":
                shrink = 0.55
            elif variant == "continuity_all":
                shrink = dynamic
            elif variant == "continuity_nonbehavior":
                shrink = 1.0 if feature in BEHAVIOR else dynamic
            else:
                raise ValueError(variant)
            row[feature] = float(lm + shrink * identity)
        output.append(row)
    return pl.DataFrame(output).sort("team_id")


def run_fold(older_season: int, latest_season: int, test_season: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame, list[dict], bool]:
    older = _canonical_policy(older_season)
    latest = _canonical_policy(latest_season)
    continuity, depth_used = _team_continuity(latest_season, test_season)
    policies = {v: _build_policy(older, latest, continuity, v) for v in VARIANTS}
    schedule = _regular_schedule(test_season)
    records: list[dict] = []
    c_rows = [{"test_season": test_season, "team": t, "continuity": continuity[t]} for t in NFL_TEAMS]

    for idx, game in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(game["away_team"])); home = normalize_team_id(str(game["home_team"]))
        paired_seed = seed + idx * 101
        for variant, policy in policies.items():
            states = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
            result = simulate_game(GameState(f"{away}@{home}-{variant}", states[away], states[home]), worlds=worlds, seed=paired_seed)
            records.append({
                "older_season": older_season, "latest_season": latest_season, "test_season": test_season,
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
            "older_season": older_season, "latest_season": latest_season, "test_season": test_season,
            "variant": variant, "games": sub.height,
            "total_bias": float((mt-at).mean()), "total_mae": float(np.abs(mt-at).mean()), "total_corr": _corr(mt, at),
            "margin_mae": float(np.abs(mm-am).mean()), "margin_corr": _corr(mm, am), "modern_depth_used": depth_used,
        })
    return metrics, frame, c_rows, depth_used


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=2026090773)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-continuity-conditioned-memory"))
    args = p.parse_args()
    folds = ((2020, 2021, 2022), (2021, 2022, 2023), (2022, 2023, 2024), (2023, 2024, 2025))
    metrics: list[dict] = []; games: list[pl.DataFrame] = []; continuity_rows: list[dict] = []; depth_flags: list[bool] = []
    for i, fold in enumerate(folds):
        m, g, c, d = run_fold(*fold, args.worlds, args.seed + i * 10_000_019)
        metrics += m; games.append(g); continuity_rows += c; depth_flags.append(d)
    mf = pl.DataFrame(metrics)
    summary = mf.group_by("variant").agg(
        pl.col("total_mae").mean().alias("mean_total_mae"),
        pl.col("total_corr").mean().alias("mean_total_corr"),
        pl.col("margin_mae").mean().alias("mean_margin_mae"),
        pl.col("margin_corr").mean().alias("mean_margin_corr"),
        pl.col("total_bias").mean().alias("mean_total_bias"),
    ).sort("mean_total_mae")
    base = summary.filter(pl.col("variant") == "latest").row(0, named=True)
    comparisons: list[dict] = []
    for variant in [v for v in VARIANTS if v != "latest"]:
        row = summary.filter(pl.col("variant") == variant).row(0, named=True)
        fv = mf.filter(pl.col("variant") == variant).sort("test_season")
        fb = mf.filter(pl.col("variant") == "latest").sort("test_season")
        tg = (fb["total_mae"] - fv["total_mae"]).to_numpy(); mg = (fb["margin_mae"] - fv["margin_mae"]).to_numpy()
        comparisons.append({**row,
            "total_mae_gain": base["mean_total_mae"] - row["mean_total_mae"],
            "margin_mae_gain": base["mean_margin_mae"] - row["mean_margin_mae"],
            "total_corr_gain": row["mean_total_corr"] - base["mean_total_corr"],
            "margin_corr_gain": row["mean_margin_corr"] - base["mean_margin_corr"],
            "positive_total_folds": int((tg > 0).sum()), "positive_margin_folds": int((mg > 0).sum()),
            "positive_both_folds": int(((tg > 0) & (mg > 0)).sum()),
        })
    cf = pl.DataFrame(comparisons).sort(["positive_both_folds", "total_mae_gain", "margin_mae_gain"], descending=True)
    best = cf.row(0, named=True)
    passed = bool(best["total_mae_gain"] > 0 and best["margin_mae_gain"] > 0 and best["total_corr_gain"] >= 0 and best["margin_corr_gain"] >= 0 and best["positive_both_folds"] >= 3)
    cdf = pl.DataFrame(continuity_rows)
    manifest = {
        "artifact": "Monster Continuity-Conditioned Memory OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "worlds_per_game_variant": args.worlds,
        "paired_common_random_numbers": True, "folds": [list(x) for x in folds],
        "continuity_mean": float(cdf["continuity"].mean()), "continuity_min": float(cdf["continuity"].min()), "continuity_max": float(cdf["continuity"].max()),
        "modern_depth_available_by_fold": depth_flags,
        "summary": summary.to_dicts(), "candidate_comparison": cf.to_dicts(), "best_candidate": best,
        "robustness_pass": passed, "market_blind": True,
        "principle": "Personnel continuity controls confidence in inherited team identity; it is not itself a positive or negative scoring bonus.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(games).write_csv(args.out / "games.csv"); mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv"); cf.write_csv(args.out / "candidate_comparison.csv")
    cdf.write_csv(args.out / "continuity.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
