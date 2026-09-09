from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _summary(values: np.ndarray) -> dict[str, float]:
    values = values.astype(float)
    return {
        "mean": float(values.mean()) if len(values) else 0.0,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p10": float(np.quantile(values, 0.10)) if len(values) else 0.0,
        "p50": float(np.quantile(values, 0.50)) if len(values) else 0.0,
        "p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/rush-role-structure"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    cols = [
        "game_id", "posteam", "rusher_player_id", "rusher_player_name", "rush_attempt",
        "qb_kneel", "qb_spike", "rusher_player_position",
    ]
    pbp = pbp.select([c for c in cols if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    rush = pbp.filter(
        (pl.col("rush_attempt").fill_null(0) == 1) & pl.col("rusher_player_id").is_not_null()
    )
    player_game = (
        rush.group_by(["game_id", "posteam", "rusher_player_id"])
        .agg(
            pl.len().alias("attempts"),
            pl.col("rusher_player_name").drop_nulls().first().alias("player") if "rusher_player_name" in rush.columns else pl.lit(None).alias("player"),
            pl.col("rusher_player_position").drop_nulls().first().alias("position") if "rusher_player_position" in rush.columns else pl.lit(None).alias("position"),
        )
        .sort(["game_id", "posteam", "attempts"], descending=[False, False, True])
        .with_columns(pl.col("attempts").rank("ordinal", descending=True).over(["game_id", "posteam"]).alias("rank"))
    )
    team = player_game.group_by(["game_id", "posteam"]).agg(
        pl.col("attempts").sum().alias("team_attempts"),
        pl.len().alias("rushers"),
        pl.col("attempts").filter(pl.col("attempts") <= 2).sum().alias("incidental_attempts"),
        (pl.col("attempts") <= 2).sum().alias("incidental_rushers"),
        pl.col("attempts").filter(pl.col("attempts") >= 3).sum().alias("core_attempts"),
        (pl.col("attempts") >= 3).sum().alias("core_rushers"),
    ).with_columns(
        (pl.col("incidental_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("incidental_share"),
        (pl.col("core_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("core_share"),
    )

    by_rank = (
        player_game.group_by("rank")
        .agg(
            pl.len().alias("observations"),
            pl.col("attempts").mean().alias("attempts_mean"),
            pl.col("attempts").median().alias("attempts_median"),
            (pl.col("attempts") <= 1).mean().alias("p_one_or_less"),
            (pl.col("attempts") <= 2).mean().alias("p_two_or_less"),
        )
        .sort("rank")
    )
    by_position = (
        player_game.with_columns(pl.col("position").fill_null("UNK"))
        .group_by("position")
        .agg(
            pl.len().alias("player_games"),
            pl.col("attempts").sum().alias("attempts"),
            pl.col("attempts").mean().alias("attempts_per_player_game"),
            (pl.col("attempts") <= 2).mean().alias("p_two_or_less"),
        )
        .sort("attempts", descending=True)
    )

    manifest = {
        "artifact": "Monster Historical Rushing Role Structure Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "team_games": int(team.height),
        "rushers_per_team_game": _summary(team["rushers"].to_numpy()),
        "core_rushers_per_team_game": _summary(team["core_rushers"].to_numpy()),
        "incidental_rushers_per_team_game": _summary(team["incidental_rushers"].to_numpy()),
        "incidental_attempts_per_team_game": _summary(team["incidental_attempts"].to_numpy()),
        "incidental_share": _summary(team["incidental_share"].to_numpy()),
        "core_share": _summary(team["core_share"].to_numpy()),
        "principle": "Estimate peripheral rushing participation from one-game attempt structure before assigning it simulation authority. Core and incidental rushing are distinct mechanisms.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    team.write_csv(args.out / "team_game_rush_structure.csv")
    player_game.write_csv(args.out / "ranked_rushers.csv")
    by_rank.write_csv(args.out / "rush_structure_by_rank.csv")
    by_position.write_csv(args.out / "rush_structure_by_position.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(by_rank.head(8))
    print(by_position)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
