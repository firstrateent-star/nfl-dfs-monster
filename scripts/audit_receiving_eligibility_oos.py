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
        pl.col("targets").rank(method="ordinal", descending=True).over(["game_id", "posteam"]).alias("rank"),
    )
    tg = (
        pg.group_by(["game_id", "posteam"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.len().alias("earners"),
            ((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_earners"),
            pl.col("targets").filter((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_targets"),
            pl.col("targets").filter(pl.col("targets") <= 2).sum().alias("peripheral_targets"),
        )
        .with_columns(
            (pl.col("middle_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("middle_share"),
            (pl.col("peripheral_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("peripheral_share"),
        )
    )
    ranks = np.array([
        float(
            ranked.filter(pl.col("rank") == rank)
            .select((pl.col("targets") / pl.col("team_targets").clip(lower_bound=1)).mean())
            .item()
        )
        for rank in range(1, RANKS + 1)
    ])
    return tg, ranks


def _observed(tg: pl.DataFrame, ranks: np.ndarray) -> dict[str, float]:
    return {
        **{f"rank{i+1}_share": float(ranks[i]) for i in range(RANKS)},
        "target_earners": float(tg["earners"].mean()),
        "middle_earners": float(tg["middle_earners"].mean()),
        "middle_share": float(tg["middle_share"].mean()),
        "peripheral_share": float(tg["peripheral_share"].mean()),
    }


def _center(rank_prior: np.ndarray, eligible: int, beta: float) -> np.ndarray:
    n = max(int(eligible), 1)
    center = np.zeros(n, dtype=float)
    top = min(RANKS, n)
    center[:top] = rank_prior[:top]
    if n > RANKS:
        tail = max(1.0 - float(rank_prior.sum()), 1e-6)
        center[RANKS:] = tail / (n - RANKS)
    center = np.power(np.clip(center, 1e-8, None), beta)
    return center / center.sum()


def _simulate(
    tg: pl.DataFrame,
    rank_prior: np.ndarray,
    *,
    beta: float,
    concentration: float,
    eligible_extra: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rank_sums = np.zeros(RANKS, dtype=float)
    earners_sum = middle_earners_sum = middle_share_sum = peripheral_share_sum = 0.0
    worlds = 0
    for row in tg.select(["team_targets", "earners"]).iter_rows(named=True):
        total = int(row["team_targets"])
        eligible = int(row["earners"]) + int(eligible_extra)
        c = _center(rank_prior, eligible, beta)
        alpha = np.clip(c * concentration, 0.15, None)
        for _ in range(replicates):
            shares = rng.dirichlet(alpha)
            counts = rng.multinomial(total, shares)
            ranked = np.sort(counts)[::-1]
            denom = max(total, 1)
            for r in range(RANKS):
                rank_sums[r] += (ranked[r] if r < len(ranked) else 0) / denom
            middle = (counts >= 3) & (counts <= 5)
            peripheral = (counts > 0) & (counts <= 2)
            earners_sum += float((counts > 0).sum())
            middle_earners_sum += float(middle.sum())
            middle_share_sum += float(counts[middle].sum()) / denom
            peripheral_share_sum += float(counts[peripheral].sum()) / denom
            worlds += 1
    return {
        **{f"rank{i+1}_share": float(rank_sums[i] / worlds) for i in range(RANKS)},
        "target_earners": float(earners_sum / worlds),
        "middle_earners": float(middle_earners_sum / worlds),
        "middle_share": float(middle_share_sum / worlds),
        "peripheral_share": float(peripheral_share_sum / worlds),
    }


def _loss(sim: dict[str, float], obs: dict[str, float]) -> float:
    terms = [
        ((sim[f"rank{r}_share"] - obs[f"rank{r}_share"]) / 0.015) ** 2
        for r in range(1, RANKS + 1)
    ]
    terms.extend([
        ((sim["target_earners"] - obs["target_earners"]) / 0.60) ** 2,
        ((sim["middle_earners"] - obs["middle_earners"]) / 0.30) ** 2,
        ((sim["middle_share"] - obs["middle_share"]) / 0.03) ** 2,
        ((sim["peripheral_share"] - obs["peripheral_share"]) / 0.03) ** 2,
    ])
    return float(np.mean(terms))


def _gate(sim: dict[str, float], obs: dict[str, float]) -> bool:
    return (
        abs(sim["rank1_share"] - obs["rank1_share"]) <= 0.04
        and abs(sim["rank2_share"] - obs["rank2_share"]) <= 0.035
        and abs(sim["rank3_share"] - obs["rank3_share"]) <= 0.03
        and abs(sim["target_earners"] - obs["target_earners"]) <= 0.75
        and abs(sim["middle_earners"] - obs["middle_earners"]) <= 0.45
        and abs(sim["middle_share"] - obs["middle_share"]) <= 0.045
        and abs(sim["peripheral_share"] - obs["peripheral_share"]) <= 0.035
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--train-window", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026090718)
    parser.add_argument("--out", type=Path, default=Path("artifacts/receiving-eligibility-oos"))
    args = parser.parse_args()

    data: dict[int, tuple[pl.DataFrame, np.ndarray]] = {season: _team_games(season) for season in args.seasons}
    beta_grid = (0.72, 0.78, 0.84, 0.90)
    concentration_grid = (100.0, 140.0, 200.0)
    extra_grid = (0, 1, 2, 3)

    fold_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for test_idx in range(args.train_window, len(args.seasons)):
        train_seasons = args.seasons[test_idx - args.train_window:test_idx]
        test_season = args.seasons[test_idx]
        train_tg = pl.concat([data[s][0] for s in train_seasons])
        train_prior = np.mean(np.vstack([data[s][1] for s in train_seasons]), axis=0)
        train_obs = _observed(train_tg, train_prior)
        test_tg, test_ranks = data[test_season]
        test_obs = _observed(test_tg, test_ranks)

        best = None
        baseline = None
        candidate_id = 0
        for beta in beta_grid:
            for concentration in concentration_grid:
                for extra in extra_grid:
                    sim = _simulate(
                        train_tg,
                        train_prior,
                        beta=beta,
                        concentration=concentration,
                        eligible_extra=extra,
                        replicates=args.replicates,
                        seed=args.seed + test_idx * 100_003 + candidate_id * 997,
                    )
                    loss = _loss(sim, train_obs)
                    row = {
                        "test_season": test_season,
                        "train_seasons": ",".join(map(str, train_seasons)),
                        "beta": beta,
                        "concentration": concentration,
                        "eligible_extra": extra,
                        "train_loss": loss,
                    }
                    candidate_rows.append(row)
                    if extra == 0 and (baseline is None or loss < baseline["train_loss"]):
                        baseline = {**row, **sim}
                    if best is None or loss < best["train_loss"]:
                        best = {**row, **sim}
                    candidate_id += 1

        assert best is not None and baseline is not None
        test_sim = _simulate(
            test_tg,
            train_prior,
            beta=float(best["beta"]),
            concentration=float(best["concentration"]),
            eligible_extra=int(best["eligible_extra"]),
            replicates=max(args.replicates * 3, 96),
            seed=args.seed + test_idx * 1_000_003 + 777,
        )
        baseline_test = _simulate(
            test_tg,
            train_prior,
            beta=float(baseline["beta"]),
            concentration=float(baseline["concentration"]),
            eligible_extra=0,
            replicates=max(args.replicates * 3, 96),
            seed=args.seed + test_idx * 1_000_003 + 888,
        )
        fold_rows.append({
            "test_season": test_season,
            "train_seasons": ",".join(map(str, train_seasons)),
            "selected_beta": best["beta"],
            "selected_concentration": best["concentration"],
            "selected_eligible_extra": best["eligible_extra"],
            "train_loss": best["train_loss"],
            "oos_loss": _loss(test_sim, test_obs),
            "baseline_oos_loss": _loss(baseline_test, test_obs),
            "oos_gate_passed": _gate(test_sim, test_obs),
            "test_target_earners": test_obs["target_earners"],
            "sim_target_earners": test_sim["target_earners"],
            "test_middle_earners": test_obs["middle_earners"],
            "sim_middle_earners": test_sim["middle_earners"],
            "test_middle_share": test_obs["middle_share"],
            "sim_middle_share": test_sim["middle_share"],
            "test_peripheral_share": test_obs["peripheral_share"],
            "sim_peripheral_share": test_sim["peripheral_share"],
            "test_rank1_share": test_obs["rank1_share"],
            "sim_rank1_share": test_sim["rank1_share"],
        })

    folds = pl.DataFrame(fold_rows)
    promotion = (
        folds.filter(pl.col("oos_gate_passed")).height >= max(3, len(fold_rows) - 1)
        and folds.filter(pl.col("oos_loss") < pl.col("baseline_oos_loss")).height >= max(3, len(fold_rows) - 1)
        and folds.select(pl.col("selected_eligible_extra").median()).item() > 0
    )

    args.out.mkdir(parents=True, exist_ok=True)
    folds.write_csv(args.out / "fold_results.csv")
    pl.DataFrame(candidate_rows).write_csv(args.out / "candidate_grid.csv")
    manifest = {
        "artifact": "Monster Receiving Eligibility Rolling OOS Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": args.seasons,
        "train_window": args.train_window,
        "fold_count": len(fold_rows),
        "folds_passing_gate": int(folds.filter(pl.col("oos_gate_passed")).height),
        "folds_beating_no_extra_baseline": int(folds.filter(pl.col("oos_loss") < pl.col("baseline_oos_loss")).height),
        "median_selected_eligible_extra": float(folds.select(pl.col("selected_eligible_extra").median()).item()),
        "eligibility_state_promoted": bool(promotion),
        "market_blind": True,
        "principle": "Latent route/read eligibility is distinct from realized target earning. Promote the extra eligibility state only if training-selected values improve multiple later seasons out of sample while preserving rank, middle-tier, and peripheral anatomy.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(folds)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
