from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.ingest.nflverse import configure_cache


def _season_metrics(frame: pl.DataFrame, season: int) -> dict:
    pg = frame.filter(pl.col("season") == season)
    team_game = (
        pg.group_by(["game_id", "posteam"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.len().alias("target_earners"),
            (pl.col("targets") <= 2).sum().alias("peripheral_earners"),
            ((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_earners"),
            (pl.col("targets") >= 6).sum().alias("high_earners"),
            pl.col("targets").filter(pl.col("targets") <= 2).sum().alias("peripheral_targets"),
            pl.col("targets").filter((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_targets"),
            pl.col("targets").filter(pl.col("targets") >= 6).sum().alias("high_targets"),
        )
        .with_columns(
            (pl.col("peripheral_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("peripheral_share"),
            (pl.col("middle_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("middle_share"),
            (pl.col("high_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("high_share"),
        )
    )

    ranked = pg.with_columns(
        pl.col("targets").rank(method="ordinal", descending=True).over(["game_id", "posteam"]).alias("rank")
    )
    rank_rows = (
        ranked.filter(pl.col("rank") <= 6)
        .group_by(["game_id", "posteam"])
        .agg([
            pl.col("targets").filter(pl.col("rank") == r).sum().alias(f"rank{r}_targets")
            for r in range(1, 7)
        ])
        .join(team_game.select(["game_id", "posteam", "team_targets"]), on=["game_id", "posteam"], how="left")
        .with_columns([
            (pl.col(f"rank{r}_targets") / pl.col("team_targets").clip(lower_bound=1)).alias(f"rank{r}_share")
            for r in range(1, 7)
        ])
    )

    result = {
        "season": season,
        "team_games": int(team_game.height),
        "team_targets_mean": float(team_game["team_targets"].mean()),
        "target_earners_mean": float(team_game["target_earners"].mean()),
        "peripheral_earners_mean": float(team_game["peripheral_earners"].mean()),
        "middle_earners_mean": float(team_game["middle_earners"].mean()),
        "high_earners_mean": float(team_game["high_earners"].mean()),
        "peripheral_share_mean": float(team_game["peripheral_share"].mean()),
        "middle_share_mean": float(team_game["middle_share"].mean()),
        "high_share_mean": float(team_game["high_share"].mean()),
    }
    for r in range(1, 7):
        result[f"rank{r}_targets_mean"] = float(rank_rows[f"rank{r}_targets"].mean())
        result[f"rank{r}_share_mean"] = float(rank_rows[f"rank{r}_share"].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/receiving-tier-stability"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp(args.seasons)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    target_play = (
        (pl.col("pass_attempt").fill_null(0) == 1)
        & pl.col("receiver_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    player_game = (
        pbp.filter(target_play)
        .group_by(["season", "game_id", "posteam", "receiver_player_id"])
        .agg(pl.len().alias("targets"))
    )

    rows = [_season_metrics(player_game, season) for season in args.seasons]
    out = pl.DataFrame(rows).sort("season")
    args.out.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out / "season_stability.csv")

    def metric_range(name: str) -> float:
        vals = out[name].to_list()
        return float(max(vals) - min(vals))

    stability = {
        "middle_earners_mean_range": metric_range("middle_earners_mean"),
        "middle_share_mean_range": metric_range("middle_share_mean"),
        "rank4_share_mean_range": metric_range("rank4_share_mean"),
        "rank5_share_mean_range": metric_range("rank5_share_mean"),
        "rank6_share_mean_range": metric_range("rank6_share_mean"),
    }
    manifest = {
        "artifact": "Monster Multi-Season Receiving Tier Stability Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": args.seasons,
        "season_metrics": rows,
        "stability_ranges": stability,
        "middle_tier_stable_candidate": bool(
            stability["middle_earners_mean_range"] <= 0.35
            and stability["middle_share_mean_range"] <= 0.04
            and stability["rank4_share_mean_range"] <= 0.025
            and stability["rank5_share_mean_range"] <= 0.02
        ),
        "market_blind": True,
        "principle": "Grant a league-level middle receiving tier prior only if rank-4+ and 3-5 target structure persists across seasons; otherwise condition it on team context.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
