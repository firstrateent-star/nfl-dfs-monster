from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

RANKS = 6


def _team_games(season: int) -> tuple[pl.DataFrame, np.ndarray]:
    import nflreadpy as nfl

    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    keep = ["game_id", "posteam", "pass_attempt", "receiver_player_id", "qb_spike"]
    pbp = pbp.select([c for c in keep if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    target_play = (
        (pl.col("pass_attempt").fill_null(0) == 1)
        & pl.col("receiver_player_id").is_not_null()
    )
    pg = (
        pbp.filter(target_play)
        .group_by(["game_id", "posteam", "receiver_player_id"])
        .agg(pl.len().alias("targets"))
    )
    ranked = pg.with_columns(
        pl.col("targets").sum().over(["game_id", "posteam"]).alias("team_targets"),
        pl.col("targets")
        .rank(method="ordinal", descending=True)
        .over(["game_id", "posteam"])
        .alias("rank"),
    )
    tg = (
        pg.group_by(["game_id", "posteam"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.len().alias("earners"),
            ((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_earners"),
            pl.col("targets").filter((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_targets"),
            (pl.col("targets") <= 2).sum().alias("peripheral_earners"),
            pl.col("targets").filter(pl.col("targets") <= 2).sum().alias("peripheral_targets"),
        )
        .with_columns(
            (pl.col("middle_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("middle_share"),
            (pl.col("peripheral_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("peripheral_share"),
        )
    )
    rank_means = np.array([
        float(
            ranked.filter(pl.col("rank") == rank)
            .select((pl.col("targets") / pl.col("team_targets").clip(lower_bound=1)).mean())
            .item()
        )
        for rank in range(1, RANKS + 1)
    ])
    return tg, rank_means


def _observed_metrics(tg: pl.DataFrame, rank_means: np.ndarray) -> dict[str, float]:
    return {
        **{f"rank{i + 1}_share": float(rank_means[i]) for i in range(RANKS)},
        "target_earners": float(tg["earners"].mean()),
        "middle_earners": float(tg["middle_earners"].mean()),
        "middle_share": float(tg["middle_share"].mean()),
        "peripheral_share": float(tg["peripheral_share"].mean()),
    }


def _latent_center(rank_prior: np.ndarray, earners: int, beta: float) -> np.ndarray:
    n = max(int(earners), 1)
    center = np.zeros(n, dtype=float)
    top = min(RANKS, n)
    center[:top] = rank_prior[:top]
    if n > RANKS:
        tail = max(1.0 - float(rank_prior.sum()), 1e-6)
        center[RANKS:] = tail / (n - RANKS)
    center = np.power(np.clip(center, 1e-8, None), beta)
    return center / center.sum()


def _simulate_metrics(
    tg: pl.DataFrame,
    rank_prior: np.ndarray,
    *,
    beta: float,
    concentration: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rank_sums = np.zeros(RANKS, dtype=float)
    middle_earners_sum = 0.0
    middle_share_sum = 0.0
    peripheral_share_sum = 0.0
    earners_sum = 0.0
    worlds = 0

    for row in tg.select(["team_targets", "earners"]).iter_rows(named=True):
        total = int(row["team_targets"])
        n = int(row["earners"])
        center = _latent_center(rank_prior, n, beta)
        alpha = np.clip(center * concentration, 0.15, None)
        for _ in range(replicates):
            shares = rng.dirichlet(alpha)
            counts = rng.multinomial(total, shares)
            ranked_counts = np.sort(counts)[::-1]
            denom = max(total, 1)
            for r in range(RANKS):
                rank_sums[r] += (ranked_counts[r] if r < len(ranked_counts) else 0) / denom
            middle = (counts >= 3) & (counts <= 5)
            peripheral = (counts > 0) & (counts <= 2)
            middle_earners_sum += float(middle.sum())
            middle_share_sum += float(counts[middle].sum()) / denom
            peripheral_share_sum += float(counts[peripheral].sum()) / denom
            earners_sum += float((counts > 0).sum())
            worlds += 1

    return {
        **{f"rank{i + 1}_share": float(rank_sums[i] / worlds) for i in range(RANKS)},
        "target_earners": float(earners_sum / worlds),
        "middle_earners": float(middle_earners_sum / worlds),
        "middle_share": float(middle_share_sum / worlds),
        "peripheral_share": float(peripheral_share_sum / worlds),
    }


def _loss(sim: dict[str, float], obs: dict[str, float]) -> float:
    # Dimensionless, interpretable tolerances. No 2025 values are used in selection.
    terms = []
    for rank in range(1, RANKS + 1):
        terms.append(((sim[f"rank{rank}_share"] - obs[f"rank{rank}_share"]) / 0.015) ** 2)
    terms.extend([
        ((sim["middle_earners"] - obs["middle_earners"]) / 0.30) ** 2,
        ((sim["middle_share"] - obs["middle_share"]) / 0.03) ** 2,
        ((sim["peripheral_share"] - obs["peripheral_share"]) / 0.03) ** 2,
        ((sim["target_earners"] - obs["target_earners"]) / 0.60) ** 2,
    ])
    return float(np.mean(terms))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--test", type=int, default=2025)
    parser.add_argument("--replicates", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026090717)
    parser.add_argument("--out", type=Path, default=Path("artifacts/receiving-oos-calibration"))
    args = parser.parse_args()

    train_frames = []
    train_rank_rows = []
    for season in args.train:
        tg, ranks = _team_games(season)
        train_frames.append(tg)
        train_rank_rows.append(ranks)
    train = pl.concat(train_frames)
    train_rank_prior = np.mean(np.vstack(train_rank_rows), axis=0)
    train_obs = _observed_metrics(train, train_rank_prior)

    test, test_ranks = _team_games(args.test)
    test_obs = _observed_metrics(test, test_ranks)

    beta_grid = (0.72, 0.78, 0.84, 0.90, 0.96, 1.02)
    concentration_grid = (70.0, 100.0, 140.0, 200.0)
    rows = []
    best = None
    for b_idx, beta in enumerate(beta_grid):
        for c_idx, concentration in enumerate(concentration_grid):
            sim = _simulate_metrics(
                train,
                train_rank_prior,
                beta=beta,
                concentration=concentration,
                replicates=args.replicates,
                seed=args.seed + b_idx * 1009 + c_idx * 9173,
            )
            loss = _loss(sim, train_obs)
            row = {"beta": beta, "concentration": concentration, "train_loss": loss, **sim}
            rows.append(row)
            if best is None or loss < best["train_loss"]:
                best = row

    assert best is not None
    test_sim = _simulate_metrics(
        test,
        train_rank_prior,
        beta=float(best["beta"]),
        concentration=float(best["concentration"]),
        replicates=max(args.replicates * 2, 96),
        seed=args.seed + 999_983,
    )

    oos_gate = (
        abs(test_sim["rank1_share"] - test_obs["rank1_share"]) <= 0.04
        and abs(test_sim["rank2_share"] - test_obs["rank2_share"]) <= 0.035
        and abs(test_sim["rank3_share"] - test_obs["rank3_share"]) <= 0.03
        and abs(test_sim["middle_earners"] - test_obs["middle_earners"]) <= 0.45
        and abs(test_sim["middle_share"] - test_obs["middle_share"]) <= 0.045
        and abs(test_sim["peripheral_share"] - test_obs["peripheral_share"]) <= 0.035
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort("train_loss").write_csv(args.out / "candidate_grid.csv")
    manifest = {
        "artifact": "Monster Receiving Latent Role Prior OOS Calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_seasons": args.train,
        "test_season": args.test,
        "selection_uses_test_season": False,
        "rank_prior_from_train_realized_means": train_rank_prior.tolist(),
        "best_train_candidate": {"beta": best["beta"], "concentration": best["concentration"], "loss": best["train_loss"]},
        "train_observed": train_obs,
        "test_observed": test_obs,
        "test_simulated": test_sim,
        "oos_gate_passed": bool(oos_gate),
        "market_blind": True,
        "principle": "Observed rank shares are realized order statistics, not latent entitlements. Learn the finite-sample correction on 2022-2024, then judge it once on held-out 2025 rather than hand-tuning Week 1 to the audit target.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pl.DataFrame(rows).sort("train_loss").head(10))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
