from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from validate_oos_offensive_personnel import VARIANTS, run_fold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=7)
    p.add_argument("--base-seed", type=int, default=2026090729)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-offensive-personnel-stress"))
    args = p.parse_args()

    seed_metrics: list[dict] = []
    for seed_idx in range(args.seeds):
        seed = args.base_seed + seed_idx * 1_000_003
        fold_rows: list[dict] = []
        for fold_idx, (train, test) in enumerate(((2023, 2024), (2024, 2025))):
            metrics, _, _ = run_fold(
                train,
                test,
                args.worlds,
                seed + fold_idx * 10_000_019,
            )
            fold_rows.extend(metrics)

        frame = pl.DataFrame(fold_rows)
        by_variant = frame.group_by("variant").agg(
            pl.col("total_mae").mean().alias("total_mae"),
            pl.col("total_corr").mean().alias("total_corr"),
            pl.col("margin_mae").mean().alias("margin_mae"),
            pl.col("margin_corr").mean().alias("margin_corr"),
        )
        policy = by_variant.filter(pl.col("variant") == "policy").row(0, named=True)
        candidate = by_variant.filter(pl.col("variant") == "continuity_offense").row(0, named=True)
        seed_metrics.append({
            "seed_index": seed_idx,
            "seed": seed,
            "policy_total_mae": policy["total_mae"],
            "candidate_total_mae": candidate["total_mae"],
            "total_mae_gain": policy["total_mae"] - candidate["total_mae"],
            "policy_margin_mae": policy["margin_mae"],
            "candidate_margin_mae": candidate["margin_mae"],
            "margin_mae_gain": policy["margin_mae"] - candidate["margin_mae"],
            "policy_total_corr": policy["total_corr"],
            "candidate_total_corr": candidate["total_corr"],
            "total_corr_gain": candidate["total_corr"] - policy["total_corr"],
            "policy_margin_corr": policy["margin_corr"],
            "candidate_margin_corr": candidate["margin_corr"],
            "margin_corr_gain": candidate["margin_corr"] - policy["margin_corr"],
        })

    sf = pl.DataFrame(seed_metrics)
    positive_total = int((sf["total_mae_gain"] > 0).sum())
    positive_margin = int((sf["margin_mae_gain"] > 0).sum())
    positive_both = int(((sf["total_mae_gain"] > 0) & (sf["margin_mae_gain"] > 0)).sum())
    positive_total_corr = int((sf["total_corr_gain"] >= 0).sum())
    positive_margin_corr = int((sf["margin_corr_gain"] >= 0).sum())

    summary = {
        "artifact": "Monster Offensive Personnel OOS Multi-Seed Stress Test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant_seed": args.worlds,
        "seed_count": args.seeds,
        "mean_total_mae_gain": float(sf["total_mae_gain"].mean()),
        "median_total_mae_gain": float(sf["total_mae_gain"].median()),
        "mean_margin_mae_gain": float(sf["margin_mae_gain"].mean()),
        "median_margin_mae_gain": float(sf["margin_mae_gain"].median()),
        "mean_total_corr_gain": float(sf["total_corr_gain"].mean()),
        "mean_margin_corr_gain": float(sf["margin_corr_gain"].mean()),
        "positive_total_mae_seeds": positive_total,
        "positive_margin_mae_seeds": positive_margin,
        "positive_both_mae_seeds": positive_both,
        "nonnegative_total_corr_seeds": positive_total_corr,
        "nonnegative_margin_corr_seeds": positive_margin_corr,
    }
    # Promotion requires directional stability, not a single-seed win.
    required = int(np.ceil(0.70 * args.seeds))
    summary["required_positive_seeds"] = required
    summary["robustness_pass"] = bool(
        summary["mean_total_mae_gain"] > 0
        and summary["mean_margin_mae_gain"] > 0
        and summary["mean_total_corr_gain"] >= 0
        and summary["mean_margin_corr_gain"] >= 0
        and positive_both >= required
        and positive_total_corr >= required
        and positive_margin_corr >= required
    )
    summary["principle"] = "A tiny OOS improvement must survive Monte Carlo seed variation before production authority is granted."

    args.out.mkdir(parents=True, exist_ok=True)
    sf.write_csv(args.out / "seed_metrics.csv")
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
