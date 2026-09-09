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

VARIANTS = ("policy", "residual25", "residual50")


def _fit_schedule_neutral_residuals(season: int, calibration_worlds: int, seed: int, ridge: float = 10.0) -> tuple[dict[str, float], dict[str, float], float]:
    """Fit persistent team offense/defense residuals after the low-level engine has spoken.

    Observation is actual team points minus the market-blind drive engine's expected
    points for each game in the completed prior season. Ridge random effects split the
    leftover residual between scoring offense and opponent defense, neutralizing schedule.
    """
    policy = _canonical_policy(season)
    schedule = _regular_schedule(season)
    teams = list(NFL_TEAMS)
    tidx = {t: i for i, t in enumerate(teams)}
    rows: list[np.ndarray] = []
    ys: list[float] = []

    for idx, game in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(game["away_team"]))
        home = normalize_team_id(str(game["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.0)
        result = simulate_game(GameState(f"cal-{season}-{away}@{home}", base[away], base[home]), worlds=calibration_worlds, seed=seed + idx * 137)
        exp_away = float(result.away_points.mean())
        exp_home = float(result.home_points.mean())
        actual_away = float(game["away_score"])
        actual_home = float(game["home_score"])

        # Columns: intercept, home effect, 32 offense effects, 32 defense effects.
        xa = np.zeros(2 + 2 * len(teams), dtype=float)
        xa[0] = 1.0
        xa[2 + tidx[away]] = 1.0
        xa[2 + len(teams) + tidx[home]] = 1.0
        rows.append(xa); ys.append(actual_away - exp_away)

        xh = np.zeros_like(xa)
        xh[0] = 1.0; xh[1] = 1.0
        xh[2 + tidx[home]] = 1.0
        xh[2 + len(teams) + tidx[away]] = 1.0
        rows.append(xh); ys.append(actual_home - exp_home)

    X = np.vstack(rows)
    y = np.asarray(ys, dtype=float)
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    penalty[1, 1] = ridge * 0.25
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    offense = {t: float(beta[2 + tidx[t]]) for t in teams}
    defense = {t: float(beta[2 + len(teams) + tidx[t]]) for t in teams}
    return offense, defense, float(beta[1])


def _apply_residual(state: TeamState, residual_points: float, shrink: float) -> TeamState:
    # Residual is only a correction to mechanisms the engine systematically missed.
    # It is not a direct score projection.
    adj = float(np.clip(shrink * residual_points / 7.0, -0.45, 0.45))
    return replace(
        state,
        td_drive_rate=float(np.clip(state.td_drive_rate * (1.0 + 0.45 * adj), 0.03, 0.55)),
        fg_drive_rate=float(np.clip(state.fg_drive_rate * (1.0 + 0.12 * adj), 0.02, 0.35)),
        turnover_drive_rate=float(np.clip(state.turnover_drive_rate * (1.0 - 0.10 * adj), 0.02, 0.35)),
        offensive_epa_per_play=float(np.clip(state.offensive_epa_per_play + 0.045 * adj, -0.45, 0.45)),
    )


def run_fold(latest_season: int, test_season: int, worlds: int, calibration_worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame, list[dict]]:
    policy = _canonical_policy(latest_season)
    off_resid, def_resid, home_resid = _fit_schedule_neutral_residuals(latest_season, calibration_worlds, seed + 5_000_003)
    schedule = _regular_schedule(test_season)
    records: list[dict] = []

    residual_rows = [
        {"latest_season": latest_season, "team": t, "offense_residual": off_resid[t], "defense_residual": def_resid[t]}
        for t in NFL_TEAMS
    ]

    for idx, game in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(game["away_team"])); home = normalize_team_id(str(game["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        paired_seed = seed + idx * 101
        away_residual = off_resid[away] + def_resid[home]
        home_residual = off_resid[home] + def_resid[away] + home_resid

        for variant in VARIANTS:
            a, h = base[away], base[home]
            if variant == "residual25":
                a = _apply_residual(a, away_residual, 0.25)
                h = _apply_residual(h, home_residual, 0.25)
            elif variant == "residual50":
                a = _apply_residual(a, away_residual, 0.50)
                h = _apply_residual(h, home_residual, 0.50)
            result = simulate_game(GameState(f"{away}@{home}-{variant}", a, h), worlds=worlds, seed=paired_seed)
            records.append({
                "latest_season": latest_season, "test_season": test_season, "variant": variant,
                "actual_total": float(game["away_score"] + game["home_score"]),
                "actual_margin": float(game["away_score"] - game["home_score"]),
                "model_total": float((result.away_points + result.home_points).mean()),
                "model_margin": float((result.away_points - result.home_points).mean()),
                "away_residual_prior": away_residual, "home_residual_prior": home_residual,
            })

    frame = pl.DataFrame(records)
    metrics: list[dict] = []
    for variant in VARIANTS:
        sub = frame.filter(pl.col("variant") == variant)
        at, mt = sub["actual_total"].to_numpy(), sub["model_total"].to_numpy()
        am, mm = sub["actual_margin"].to_numpy(), sub["model_margin"].to_numpy()
        metrics.append({
            "latest_season": latest_season, "test_season": test_season, "variant": variant, "games": sub.height,
            "total_bias": float((mt-at).mean()), "total_mae": float(np.abs(mt-at).mean()), "total_corr": _corr(mt, at),
            "margin_mae": float(np.abs(mm-am).mean()), "margin_corr": _corr(mm, am),
        })
    return metrics, frame, residual_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2000)
    p.add_argument("--calibration-worlds", type=int, default=700)
    p.add_argument("--seed", type=int, default=2026090767)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-schedule-neutral-residual"))
    args = p.parse_args()
    folds = ((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))
    metrics: list[dict] = []; games: list[pl.DataFrame] = []; residuals: list[dict] = []
    for i, fold in enumerate(folds):
        m, g, r = run_fold(*fold, args.worlds, args.calibration_worlds, args.seed + i * 10_000_019)
        metrics += m; games.append(g); residuals += r

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
        "artifact": "Monster Schedule-Neutral Residual OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds, "calibration_worlds": args.calibration_worlds,
        "paired_common_random_numbers": True, "folds": [list(x) for x in folds],
        "summary": summary.to_dicts(), "candidate_comparison": cf.to_dicts(), "best_candidate": best,
        "robustness_pass": passed, "market_blind": True,
        "principle": "Correct only persistent schedule-neutral residuals left after the causal drive engine; do not double-count raw scoreboard strength.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(games).write_csv(args.out / "games.csv")
    mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    cf.write_csv(args.out / "candidate_comparison.csv")
    pl.DataFrame(residuals).write_csv(args.out / "team_residuals.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
