from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from validate_oos_regressed_team_state import _canonical_policy, _corr, _regular_schedule
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState, TeamState
from monster.teams import NFL_TEAMS, normalize_team_id

VARIANTS = ("policy", "score_strength50", "score_strength70", "score_strength_multiyear")


def _team_score_table(season: int) -> tuple[dict[str, float], dict[str, float], float]:
    schedule = _regular_schedule(season)
    totals: dict[str, list[float]] = {t: [] for t in NFL_TEAMS}
    allowed: dict[str, list[float]] = {t: [] for t in NFL_TEAMS}
    all_scores: list[float] = []
    for g in schedule.to_dicts():
        away = normalize_team_id(str(g["away_team"]))
        home = normalize_team_id(str(g["home_team"]))
        a = float(g["away_score"]); h = float(g["home_score"])
        totals[away].append(a); totals[home].append(h)
        allowed[away].append(h); allowed[home].append(a)
        all_scores.extend([a, h])
    pf = {t: float(np.mean(v)) if v else float(np.mean(all_scores)) for t, v in totals.items()}
    pa = {t: float(np.mean(v)) if v else float(np.mean(all_scores)) for t, v in allowed.items()}
    return pf, pa, float(np.mean(all_scores))


def _score_strength(
    team: str,
    opponent: str,
    latest_pf: dict[str, float],
    latest_pa: dict[str, float],
    league_mean: float,
    shrink: float,
    older_pf: dict[str, float] | None = None,
    older_pa: dict[str, float] | None = None,
    older_mean: float | None = None,
) -> float:
    off_dev = latest_pf[team] - league_mean
    def_dev = latest_pa[opponent] - league_mean
    if older_pf is not None and older_pa is not None and older_mean is not None:
        off_dev = 0.75 * off_dev + 0.25 * (older_pf[team] - older_mean)
        def_dev = 0.75 * def_dev + 0.25 * (older_pa[opponent] - older_mean)
    expected_delta = shrink * (0.55 * off_dev + 0.45 * def_dev)
    return float(np.clip(expected_delta / max(league_mean, 1.0), -0.30, 0.30))


def _apply_score_strength(state: TeamState, strength: float) -> TeamState:
    # High-level historical scoring strength modifies conversion anatomy rather than
    # replacing the drive simulation with a direct point projection.
    return replace(
        state,
        td_drive_rate=float(np.clip(state.td_drive_rate * (1.0 + 0.80 * strength), 0.03, 0.55)),
        fg_drive_rate=float(np.clip(state.fg_drive_rate * (1.0 + 0.30 * strength), 0.02, 0.35)),
        turnover_drive_rate=float(np.clip(state.turnover_drive_rate * (1.0 - 0.25 * strength), 0.02, 0.35)),
        offensive_epa_per_play=float(np.clip(state.offensive_epa_per_play + 0.10 * strength, -0.45, 0.45)),
    )


def run_fold(older_season: int, latest_season: int, test_season: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    policy = _canonical_policy(latest_season)
    latest_pf, latest_pa, latest_mean = _team_score_table(latest_season)
    older_pf, older_pa, older_mean = _team_score_table(older_season)
    schedule = _regular_schedule(test_season)
    records: list[dict] = []

    for idx, game in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(game["away_team"])); home = normalize_team_id(str(game["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        paired_seed = seed + idx * 101
        for variant in VARIANTS:
            a, h = base[away], base[home]
            if variant == "score_strength50":
                a = _apply_score_strength(a, _score_strength(away, home, latest_pf, latest_pa, latest_mean, 0.50))
                h = _apply_score_strength(h, _score_strength(home, away, latest_pf, latest_pa, latest_mean, 0.50))
            elif variant == "score_strength70":
                a = _apply_score_strength(a, _score_strength(away, home, latest_pf, latest_pa, latest_mean, 0.70))
                h = _apply_score_strength(h, _score_strength(home, away, latest_pf, latest_pa, latest_mean, 0.70))
            elif variant == "score_strength_multiyear":
                a = _apply_score_strength(a, _score_strength(away, home, latest_pf, latest_pa, latest_mean, 0.60, older_pf, older_pa, older_mean))
                h = _apply_score_strength(h, _score_strength(home, away, latest_pf, latest_pa, latest_mean, 0.60, older_pf, older_pa, older_mean))
            result = simulate_game(GameState(f"{away}@{home}-{variant}", a, h), worlds=worlds, seed=paired_seed)
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
            "margin_mae": float(np.abs(mm-am).mean()), "margin_corr": _corr(mm, am),
        })
    return metrics, frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2500)
    p.add_argument("--seed", type=int, default=2026090759)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-score-strength-prior"))
    args = p.parse_args()
    folds = ((2020, 2021, 2022), (2021, 2022, 2023), (2022, 2023, 2024), (2023, 2024, 2025))
    metrics: list[dict] = []; games: list[pl.DataFrame] = []
    for i, fold in enumerate(folds):
        m, g = run_fold(*fold, args.worlds, args.seed + i * 10_000_019); metrics += m; games.append(g)
    mf = pl.DataFrame(metrics)
    summary = mf.group_by("variant").agg(
        pl.col("total_mae").mean().alias("mean_total_mae"),
        pl.col("total_corr").mean().alias("mean_total_corr"),
        pl.col("margin_mae").mean().alias("mean_margin_mae"),
        pl.col("margin_corr").mean().alias("mean_margin_corr"),
        pl.col("total_bias").mean().alias("mean_total_bias"),
    ).sort("mean_total_mae")
    base = summary.filter(pl.col("variant") == "policy").row(0, named=True)
    comparisons: list[dict] = []
    for variant in [v for v in VARIANTS if v != "policy"]:
        row = summary.filter(pl.col("variant") == variant).row(0, named=True)
        fv = mf.filter(pl.col("variant") == variant).sort("test_season")
        fb = mf.filter(pl.col("variant") == "policy").sort("test_season")
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
    manifest = {
        "artifact": "Monster Historical Score-Strength Prior OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "worlds_per_game_variant": args.worlds,
        "paired_common_random_numbers": True, "folds": [list(x) for x in folds],
        "summary": summary.to_dicts(), "candidate_comparison": cf.to_dicts(), "best_candidate": best,
        "robustness_pass": passed, "market_blind": True,
        "principle": "Historical scoreboard production is a market-blind hierarchical prior on scoring capacity; the drive engine remains responsible for causal score anatomy.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(games).write_csv(args.out / "games.csv"); mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv"); cf.write_csv(args.out / "candidate_comparison.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
