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
    if len(values) == 0:
        return {"mean": 0.0, "sd": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _season_frame(pbp: pl.DataFrame, season: int) -> tuple[pl.DataFrame, dict]:
    frame = pbp.filter(pl.col("season") == season) if "season" in pbp.columns else pbp
    if "season_type" in frame.columns:
        frame = frame.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in frame.columns:
        frame = frame.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in frame.columns:
        frame = frame.filter(pl.col("qb_spike").fill_null(0) == 0)

    rush = frame.filter(
        (pl.col("rush_attempt").fill_null(0) == 1)
        & pl.col("rusher_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    player_game = (
        rush.group_by(["game_id", "posteam", "rusher_player_id"])
        .agg(pl.len().alias("attempts"))
        .sort(["game_id", "posteam", "attempts"], descending=[False, False, True])
        .with_columns(
            pl.col("attempts")
            .rank("ordinal", descending=True)
            .over(["game_id", "posteam"])
            .alias("rank")
        )
    )
    team = player_game.group_by(["game_id", "posteam"]).agg(
        pl.col("attempts").sum().alias("team_attempts"),
        pl.len().alias("rushers"),
        (pl.col("attempts") >= 3).sum().alias("core_rushers"),
        ((pl.col("attempts") > 0) & (pl.col("attempts") <= 2)).sum().alias("incidental_rushers"),
        pl.col("attempts").filter(pl.col("attempts") <= 2).sum().alias("incidental_attempts"),
    ).with_columns(
        (pl.col("incidental_attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("incidental_share")
    )
    ranked = player_game.join(
        team.select("game_id", "posteam", "team_attempts"),
        on=["game_id", "posteam"], how="left"
    ).with_columns(
        (pl.col("attempts") / pl.col("team_attempts").clip(lower_bound=1)).alias("share")
    )

    metrics: dict[str, object] = {
        "season": season,
        "team_games": int(team.height),
        "team_attempts": _summary(team["team_attempts"].to_numpy()),
        "rushers": _summary(team["rushers"].to_numpy()),
        "core_rushers": _summary(team["core_rushers"].to_numpy()),
        "incidental_rushers": _summary(team["incidental_rushers"].to_numpy()),
        "incidental_share": _summary(team["incidental_share"].to_numpy()),
    }
    for rank in (1, 2, 3):
        sub = ranked.filter(pl.col("rank") == rank)
        metrics[f"rank{rank}_attempts"] = _summary(sub["attempts"].to_numpy())
        metrics[f"rank{rank}_share"] = _summary(sub["share"].to_numpy())
    return ranked.with_columns(pl.lit(season).alias("season")), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/rush-hierarchy-stability"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp(args.seasons)
    keep = [
        "season", "season_type", "game_id", "posteam", "rush_attempt",
        "rusher_player_id", "qb_kneel", "qb_spike",
    ]
    pbp = pbp.select([c for c in keep if c in pbp.columns])

    ranked_frames = []
    season_metrics = []
    for season in args.seasons:
        ranked, metrics = _season_frame(pbp, season)
        ranked_frames.append(ranked)
        season_metrics.append(metrics)

    rows = []
    for item in season_metrics:
        row = {"season": item["season"], "team_games": item["team_games"]}
        for metric in (
            "team_attempts", "rushers", "core_rushers", "incidental_rushers", "incidental_share",
            "rank1_attempts", "rank1_share", "rank2_attempts", "rank2_share", "rank3_attempts", "rank3_share",
        ):
            row[f"{metric}_mean"] = item[metric]["mean"]
            row[f"{metric}_p50"] = item[metric]["p50"]
        rows.append(row)
    stability = pl.DataFrame(rows).sort("season")

    share_cols = ["rank1_share_mean", "rank2_share_mean", "rank3_share_mean"]
    ranges = {col: float(stability[col].max() - stability[col].min()) for col in share_cols}
    structural_ranges = {
        "core_rushers_mean_range": float(stability["core_rushers_mean"].max() - stability["core_rushers_mean"].min()),
        "incidental_share_mean_range": float(stability["incidental_share_mean"].max() - stability["incidental_share_mean"].min()),
    }
    hierarchy_stable = (
        ranges["rank1_share_mean"] <= 0.035
        and ranges["rank2_share_mean"] <= 0.025
        and ranges["rank3_share_mean"] <= 0.020
        and structural_ranges["core_rushers_mean_range"] <= 0.20
        and structural_ranges["incidental_share_mean_range"] <= 0.025
    )

    pooled = pl.concat(ranked_frames)
    pooled_metrics = {}
    for rank in (1, 2, 3):
        sub = pooled.filter(pl.col("rank") == rank)
        pooled_metrics[f"rank{rank}_share"] = _summary(sub["share"].to_numpy())
        pooled_metrics[f"rank{rank}_attempts"] = _summary(sub["attempts"].to_numpy())

    args.out.mkdir(parents=True, exist_ok=True)
    stability.write_csv(args.out / "season_stability.csv")
    pooled.write_csv(args.out / "pooled_ranked_rushers.csv")
    manifest = {
        "artifact": "Monster Multi-Season Rushing Hierarchy Stability Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": args.seasons,
        "season_metrics": season_metrics,
        "mean_share_ranges": ranges,
        "structural_ranges": structural_ranges,
        "hierarchy_stable_candidate": bool(hierarchy_stable),
        "pooled": pooled_metrics,
        "market_blind": True,
        "principle": "Only grant league-wide Role A/B/C rushing hierarchy authority if ranked game-level structure remains stable across seasons; otherwise condition the hierarchy on team context.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(stability)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
