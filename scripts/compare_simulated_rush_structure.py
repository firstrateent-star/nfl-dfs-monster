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
    parser.add_argument("--out", type=Path, default=Path("artifacts/simulated-rush-structure"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    keep = ["game_id", "posteam", "rush_attempt", "rusher_player_id", "qb_kneel", "qb_spike"]
    pbp = pbp.select([c for c in keep if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    player_game = (
        pbp.filter(
            (pl.col("rush_attempt").fill_null(0) == 1)
            & pl.col("rusher_player_id").is_not_null()
        )
        .group_by(["game_id", "posteam", "rusher_player_id"])
        .agg(pl.len().alias("attempts"))
    )

    ranked = player_game.with_columns(
        pl.col("attempts")
        .rank(method="ordinal", descending=True)
        .over(["game_id", "posteam"])
        .alias("rank")
    )
    rank_pivot = (
        ranked.filter(pl.col("rank") <= 3)
        .group_by(["game_id", "posteam"])
        .agg(
            pl.col("attempts").filter(pl.col("rank") == 1).sum().alias("rank1_attempts"),
            pl.col("attempts").filter(pl.col("rank") == 2).sum().alias("rank2_attempts"),
            pl.col("attempts").filter(pl.col("rank") == 3).sum().alias("rank3_attempts"),
        )
    )

    hist = (
        player_game.group_by(["game_id", "posteam"])
        .agg(
            pl.len().alias("rushers"),
            (pl.col("attempts") >= 3).sum().alias("core_rushers"),
            ((pl.col("attempts") > 0) & (pl.col("attempts") <= 2)).sum().alias("incidental_rushers"),
            pl.col("attempts").filter(pl.col("attempts") <= 2).sum().alias("incidental_attempts"),
            pl.col("attempts").sum().alias("team_attempts"),
        )
        .join(rank_pivot, on=["game_id", "posteam"], how="left")
        .fill_null(0)
        .with_columns(
            (pl.col("incidental_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("incidental_share"),
            (pl.col("rank1_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("rank1_share"),
            (pl.col("rank2_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("rank2_share"),
            (pl.col("rank3_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("rank3_share"),
        )
    )

    sim = _read(args.team_opportunity)
    historical = {
        "rushers": float(hist["rushers"].mean()),
        "core_rushers": float(hist["core_rushers"].mean()),
        "incidental_rushers": float(hist["incidental_rushers"].mean()),
        "incidental_attempts": float(hist["incidental_attempts"].mean()),
        "incidental_share": float(hist["incidental_share"].mean()),
        "rank1_attempts": float(hist["rank1_attempts"].mean()),
        "rank2_attempts": float(hist["rank2_attempts"].mean()),
        "rank3_attempts": float(hist["rank3_attempts"].mean()),
        "rank1_share": float(hist["rank1_share"].mean()),
        "rank2_share": float(hist["rank2_share"].mean()),
        "rank3_share": float(hist["rank3_share"].mean()),
    }
    simulated = {
        "rushers": float(sim["rushers_per_world_mean"].mean()),
        "core_rushers": float(sim["core_rushers_per_world_mean"].mean()),
        "incidental_rushers": float(sim["incidental_rushers_per_world_mean"].mean()),
        "incidental_attempts": float(sim["incidental_rush_attempts_mean"].mean()),
        "incidental_share": float(sim["incidental_rush_share_mean"].mean()),
        "rank1_attempts": float(sim["rush_rank1_attempts_world_mean"].mean()),
        "rank2_attempts": float(sim["rush_rank2_attempts_world_mean"].mean()),
        "rank3_attempts": float(sim["rush_rank3_attempts_world_mean"].mean()),
        "rank1_share": float(sim["rush_rank1_share_world_mean"].mean()),
        "rank2_share": float(sim["rush_rank2_share_world_mean"].mean()),
        "rank3_share": float(sim["rush_rank3_share_world_mean"].mean()),
    }
    rows = []
    for metric in historical:
        rows.append({
            "metric": metric,
            "historical": historical[metric],
            "simulated": simulated[metric],
            "delta": simulated[metric] - historical[metric],
        })
    comparison = pl.DataFrame(rows)

    outer_structure_within_candidate_band = (
        abs(simulated["rushers"] - historical["rushers"]) <= 0.50
        and abs(simulated["core_rushers"] - historical["core_rushers"]) <= 0.30
        and abs(simulated["incidental_rushers"] - historical["incidental_rushers"]) <= 0.35
        and abs(simulated["incidental_share"] - historical["incidental_share"]) <= 0.025
    )
    realized_rank_structure_within_candidate_band = (
        abs(simulated["rank1_share"] - historical["rank1_share"]) <= 0.05
        and abs(simulated["rank2_share"] - historical["rank2_share"]) <= 0.04
        and abs(simulated["rank3_share"] - historical["rank3_share"]) <= 0.03
    )

    args.out.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(args.out / "rush_structure_comparison.csv")
    manifest = {
        "artifact": "Monster Simulated vs Historical Rushing Role Structure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "historical_team_games": int(hist.height),
        "simulated_team_games": int(sim.height),
        "historical": historical,
        "simulated": simulated,
        "outer_structure_within_candidate_band": bool(outer_structure_within_candidate_band),
        "realized_rank_structure_within_candidate_band": bool(realized_rank_structure_within_candidate_band),
        "market_blind": True,
        "principle": "Compare realized historical game ranks to ranks inside each simulated world. Cross-world player means answer a different epistemic question and must not be used to tune one-game football hierarchy.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
