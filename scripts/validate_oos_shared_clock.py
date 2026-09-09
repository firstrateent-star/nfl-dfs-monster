from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.sim.game import (
    GameWorlds,
    _drive_expectation,
    _mean_one_lognormal,
    _simulate_team_drives,
    simulate_game,
)
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import normalize_team_id
from validate_oos_continuity_conditioned_memory import _build_policy, _team_continuity
from validate_oos_regressed_team_state import _canonical_policy, _corr, _regular_schedule

ENGINES = ("legacy_independent", "shared_clock")
FOLDS = ((2020, 2021, 2022), (2021, 2022, 2023), (2022, 2023, 2024), (2023, 2024, 2025))

# Preregistered non-inferiority tolerances. These are deliberately small relative to an NFL score:
# a clock-realism correction may be retained if it does not worsen average MAE by more than
# 0.20 points or correlation by more than 0.01, while materially fixing possession/play shape.
MAE_TOLERANCE = 0.20
CORR_TOLERANCE = 0.01


def _legacy_independent_game(game: GameState, worlds: int, seed: int) -> GameWorlds:
    """Historical control: each offense receives an independent Poisson possession lottery.

    Scoring conditional on the sampled drives uses the current scoring kernel. This isolates the
    possession-clock architecture rather than mixing in unrelated score-model changes.
    """
    rng = np.random.default_rng(seed)
    shared = _mean_one_lognormal(rng, 0.08 if game.dome else 0.11, worlds)

    away_mu = _drive_expectation(game.away, game.home, False)
    home_mu = _drive_expectation(game.home, game.away, True)
    away_drives = rng.poisson(np.clip(away_mu * shared, 6.0, 16.0)).astype(np.int16)
    home_drives = rng.poisson(np.clip(home_mu * shared, 6.0, 16.0)).astype(np.int16)

    away = _simulate_team_drives(rng, game.away, game.home, away_drives, worlds)
    home = _simulate_team_drives(rng, game.home, game.away, home_drives, worlds)
    return GameWorlds(
        away_points=away[0],
        home_points=home[0],
        away_drives=away_drives,
        home_drives=home_drives,
        away_touchdowns=away[1],
        home_touchdowns=home[1],
        away_field_goals=away[2],
        home_field_goals=home[2],
        away_turnovers=away[3],
        home_turnovers=home[3],
        away_pass_disruption=away[4],
        home_pass_disruption=home[4],
        away_run_efficiency=away[5],
        home_run_efficiency=home[5],
    )


def _metrics(frame: pl.DataFrame, engine: str, test_season: int) -> dict[str, float | int | str]:
    sub = frame.filter((pl.col("engine") == engine) & (pl.col("test_season") == test_season))
    actual_total = sub["actual_total"].to_numpy()
    model_total = sub["model_total"].to_numpy()
    actual_margin = sub["actual_margin"].to_numpy()
    model_margin = sub["model_margin"].to_numpy()
    return {
        "test_season": test_season,
        "engine": engine,
        "games": sub.height,
        "total_bias": float((model_total - actual_total).mean()),
        "total_mae": float(np.abs(model_total - actual_total).mean()),
        "total_corr": _corr(model_total, actual_total),
        "margin_mae": float(np.abs(model_margin - actual_margin).mean()),
        "margin_corr": _corr(model_margin, actual_margin),
        "mean_team_drives": float(
            np.concatenate([sub["away_drives_mean"].to_numpy(), sub["home_drives_mean"].to_numpy()]).mean()
        ),
        "mean_game_drive_imbalance": float(sub["drive_imbalance_mean"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026090791)
    parser.add_argument("--out", type=Path, default=Path("artifacts/oos-shared-clock"))
    args = parser.parse_args()

    records: list[dict] = []
    depth_flags: list[bool] = []
    for fold_idx, (older_season, latest_season, test_season) in enumerate(FOLDS):
        older = _canonical_policy(older_season)
        latest = _canonical_policy(latest_season)
        continuity, depth_used = _team_continuity(latest_season, test_season)
        depth_flags.append(depth_used)
        policy = _build_policy(older, latest, continuity, "continuity_all")
        schedule = _regular_schedule(test_season)

        for game_idx, game_row in enumerate(schedule.to_dicts()):
            away = normalize_team_id(str(game_row["away_team"]))
            home = normalize_team_id(str(game_row["home_team"]))
            states = compile_team_state_map(
                policy,
                {away: home, home: away},
                prior_uncertainty=0.16,
            )
            game = GameState(f"{away}@{home}", states[away], states[home])
            paired_seed = args.seed + fold_idx * 10_000_019 + game_idx * 101

            for engine in ENGINES:
                if engine == "legacy_independent":
                    result = _legacy_independent_game(game, args.worlds, paired_seed)
                else:
                    result = simulate_game(game, args.worlds, paired_seed)
                imbalance = np.abs(result.away_drives.astype(float) - result.home_drives.astype(float))
                records.append({
                    "older_season": older_season,
                    "latest_season": latest_season,
                    "test_season": test_season,
                    "engine": engine,
                    "game": f"{away}@{home}",
                    "actual_total": float(game_row["away_score"] + game_row["home_score"]),
                    "actual_margin": float(game_row["away_score"] - game_row["home_score"]),
                    "model_total": float((result.away_points + result.home_points).mean()),
                    "model_margin": float((result.away_points - result.home_points).mean()),
                    "away_drives_mean": float(result.away_drives.mean()),
                    "home_drives_mean": float(result.home_drives.mean()),
                    "game_total_drives_mean": float((result.away_drives + result.home_drives).mean()),
                    "drive_imbalance_mean": float(imbalance.mean()),
                    "drive_imbalance_p90": float(np.quantile(imbalance, 0.90)),
                })

    games = pl.DataFrame(records)
    fold_rows = [
        _metrics(games, engine, test_season)
        for test_season in [2022, 2023, 2024, 2025]
        for engine in ENGINES
    ]
    fold_metrics = pl.DataFrame(fold_rows)

    summary = (
        fold_metrics.group_by("engine")
        .agg(
            pl.col("total_mae").mean().alias("mean_total_mae"),
            pl.col("margin_mae").mean().alias("mean_margin_mae"),
            pl.col("total_corr").mean().alias("mean_total_corr"),
            pl.col("margin_corr").mean().alias("mean_margin_corr"),
            pl.col("total_bias").mean().alias("mean_total_bias"),
            pl.col("mean_team_drives").mean().alias("mean_team_drives"),
            pl.col("mean_game_drive_imbalance").mean().alias("mean_game_drive_imbalance"),
        )
        .sort("engine")
    )

    legacy = summary.filter(pl.col("engine") == "legacy_independent").row(0, named=True)
    shared = summary.filter(pl.col("engine") == "shared_clock").row(0, named=True)
    paired = (
        fold_metrics.filter(pl.col("engine") == "shared_clock")
        .sort("test_season")
        .join(
            fold_metrics.filter(pl.col("engine") == "legacy_independent")
            .sort("test_season")
            .select(
                "test_season",
                pl.col("total_mae").alias("legacy_total_mae"),
                pl.col("margin_mae").alias("legacy_margin_mae"),
            ),
            on="test_season",
        )
        .with_columns(
            (pl.col("legacy_total_mae") - pl.col("total_mae")).alias("shared_total_mae_gain"),
            (pl.col("legacy_margin_mae") - pl.col("margin_mae")).alias("shared_margin_mae_gain"),
        )
    )

    total_mae_delta = float(shared["mean_total_mae"] - legacy["mean_total_mae"])
    margin_mae_delta = float(shared["mean_margin_mae"] - legacy["mean_margin_mae"])
    total_corr_delta = float(shared["mean_total_corr"] - legacy["mean_total_corr"])
    margin_corr_delta = float(shared["mean_margin_corr"] - legacy["mean_margin_corr"])
    positive_total_folds = int((paired["shared_total_mae_gain"] >= 0).sum())
    positive_margin_folds = int((paired["shared_margin_mae_gain"] >= 0).sum())

    noninferior = bool(
        total_mae_delta <= MAE_TOLERANCE
        and margin_mae_delta <= MAE_TOLERANCE
        and total_corr_delta >= -CORR_TOLERANCE
        and margin_corr_delta >= -CORR_TOLERANCE
    )
    clock_realism_pass = bool(
        float(shared["mean_game_drive_imbalance"]) <= 1.0
        and float(legacy["mean_game_drive_imbalance"]) > 1.0
    )
    promotion_pass = bool(noninferior and clock_realism_pass)

    manifest = {
        "artifact": "Monster Shared Clock OOS Engine A/B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_engine": args.worlds,
        "matched_seed_by_game": True,
        "folds": [list(x) for x in FOLDS],
        "policy": "continuity_all",
        "market_blind": True,
        "preregistered_tolerances": {
            "mae_points": MAE_TOLERANCE,
            "correlation": CORR_TOLERANCE,
        },
        "summary": summary.to_dicts(),
        "shared_minus_legacy": {
            "total_mae": total_mae_delta,
            "margin_mae": margin_mae_delta,
            "total_corr": total_corr_delta,
            "margin_corr": margin_corr_delta,
        },
        "positive_or_equal_total_folds": positive_total_folds,
        "positive_or_equal_margin_folds": positive_margin_folds,
        "scoring_noninferiority_pass": noninferior,
        "clock_realism_pass": clock_realism_pass,
        "promotion_pass": promotion_pass,
        "modern_depth_available_by_fold": depth_flags,
        "principle": "A more realistic clock is promoted only if it preserves predictive scoring authority out of sample.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "games.csv")
    fold_metrics.write_csv(args.out / "fold_metrics.csv")
    paired.write_csv(args.out / "paired_fold_comparison.csv")
    summary.write_csv(args.out / "summary.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
