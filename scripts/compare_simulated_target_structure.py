from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.ingest.nflverse import configure_cache


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-opportunity", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/simulated-target-structure"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    keep = ["game_id", "posteam", "pass_attempt", "receiver_player_id", "sack", "qb_spike"]
    pbp = pbp.select([c for c in keep if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    target_play = (
        (pl.col("pass_attempt").fill_null(0) == 1)
        & pl.col("receiver_player_id").is_not_null()
    )
    player_game = (
        pbp.filter(target_play)
        .group_by(["game_id", "posteam", "receiver_player_id"])
        .agg(pl.len().alias("targets"))
    )
    ranked = player_game.with_columns(
        pl.col("targets")
        .rank(method="ordinal", descending=True)
        .over(["game_id", "posteam"])
        .alias("rank")
    )
    rank_pivot = (
        ranked.filter(pl.col("rank") <= 3)
        .group_by(["game_id", "posteam"])
        .agg(
            pl.col("targets").filter(pl.col("rank") == 1).sum().alias("rank1_targets"),
            pl.col("targets").filter(pl.col("rank") == 2).sum().alias("rank2_targets"),
            pl.col("targets").filter(pl.col("rank") == 3).sum().alias("rank3_targets"),
        )
    )
    hist = (
        player_game.group_by(["game_id", "posteam"])
        .agg(
            pl.len().alias("target_earners"),
            pl.col("targets").sum().alias("team_targets"),
            (pl.col("targets") == 1).sum().alias("one_target_earners"),
            (pl.col("targets") <= 2).sum().alias("one_two_target_earners"),
            pl.col("targets").filter(pl.col("targets") <= 2).sum().alias("peripheral_targets"),
        )
        .join(rank_pivot, on=["game_id", "posteam"], how="left")
        .fill_null(0)
        .with_columns(
            (pl.col("rank1_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("rank1_share"),
            (pl.col("rank2_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("rank2_share"),
            (pl.col("rank3_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("rank3_share"),
            (pl.col("peripheral_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("peripheral_share"),
        )
    )

    sim = _read(args.team_opportunity)
    historical = {
        "target_earners": float(hist["target_earners"].mean()),
        "rank1_targets": float(hist["rank1_targets"].mean()),
        "rank2_targets": float(hist["rank2_targets"].mean()),
        "rank3_targets": float(hist["rank3_targets"].mean()),
        "rank1_share": float(hist["rank1_share"].mean()),
        "rank2_share": float(hist["rank2_share"].mean()),
        "rank3_share": float(hist["rank3_share"].mean()),
        "one_target_earners": float(hist["one_target_earners"].mean()),
        "one_two_target_earners": float(hist["one_two_target_earners"].mean()),
        "peripheral_targets": float(hist["peripheral_targets"].mean()),
        "peripheral_share": float(hist["peripheral_share"].mean()),
    }
    simulated = {
        "target_earners": float(sim["target_earners_per_world_mean"].mean()),
        "rank1_targets": float(sim["target_rank1_attempts_world_mean"].mean()),
        "rank2_targets": float(sim["target_rank2_attempts_world_mean"].mean()),
        "rank3_targets": float(sim["target_rank3_attempts_world_mean"].mean()),
        "rank1_share": float(sim["target_rank1_share_world_mean"].mean()),
        "rank2_share": float(sim["target_rank2_share_world_mean"].mean()),
        "rank3_share": float(sim["target_rank3_share_world_mean"].mean()),
    }

    rows = []
    for metric in ("target_earners", "rank1_targets", "rank2_targets", "rank3_targets", "rank1_share", "rank2_share", "rank3_share"):
        rows.append({
            "metric": metric,
            "historical": historical[metric],
            "simulated": simulated[metric],
            "delta": simulated[metric] - historical[metric],
        })
    comparison = pl.DataFrame(rows)
    realized_rank_structure_within_candidate_band = (
        abs(simulated["rank1_share"] - historical["rank1_share"]) <= 0.04
        and abs(simulated["rank2_share"] - historical["rank2_share"]) <= 0.035
        and abs(simulated["rank3_share"] - historical["rank3_share"]) <= 0.03
    )
    breadth_within_candidate_band = abs(simulated["target_earners"] - historical["target_earners"]) <= 0.75

    args.out.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(args.out / "target_structure_comparison.csv")
    manifest = {
        "artifact": "Monster Simulated vs Historical Target Role Structure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "historical_team_games": int(hist.height),
        "simulated_team_games": int(sim.height),
        "historical": historical,
        "simulated": simulated,
        "realized_rank_structure_within_candidate_band": bool(realized_rank_structure_within_candidate_band),
        "breadth_within_candidate_band": bool(breadth_within_candidate_band),
        "market_blind": True,
        "principle": "Validate receiving anatomy inside simulated worlds before tuning cross-world player projection concentration. Realized rank structure and role-identity uncertainty are separate objects.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
