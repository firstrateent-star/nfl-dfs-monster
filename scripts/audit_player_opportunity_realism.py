from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(float), q))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = values.astype(float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p10": _q(values, 0.10),
        "p50": _q(values, 0.50),
        "p90": _q(values, 0.90),
    }


def _ranked_team_game(frame: pl.DataFrame, count_col: str, player_col: str) -> pl.DataFrame:
    base = (
        frame.filter(pl.col(player_col).is_not_null())
        .group_by(["game_id", "posteam", player_col])
        .agg(pl.sum(count_col).alias("opps"))
        .filter(pl.col("opps") > 0)
    )
    team = base.group_by(["game_id", "posteam"]).agg(
        pl.col("opps").sum().alias("team_opps"),
        pl.len().alias("players_with_opps"),
    )
    return (
        base.sort(["game_id", "posteam", "opps"], descending=[False, False, True])
        .with_columns(pl.col("opps").rank("ordinal", descending=True).over(["game_id", "posteam"]).alias("rank"))
        .join(team, on=["game_id", "posteam"], how="left")
        .with_columns((pl.col("opps") / pl.col("team_opps").clip(lower_bound=1)).alias("share"))
    )


def _historical_rank_metrics(ranked: pl.DataFrame, prefix: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for rank in (1, 2, 3):
        sub = ranked.filter(pl.col("rank") == rank)
        output[f"{prefix}_rank{rank}_count"] = _summary(sub["opps"].to_numpy())
        output[f"{prefix}_rank{rank}_share"] = _summary(sub["share"].to_numpy())
    leaders = ranked.filter(pl.col("rank") == 1)
    output[f"{prefix}_players_with_opps"] = _summary(leaders["players_with_opps"].to_numpy())
    output[f"{prefix}_team_opps"] = _summary(leaders["team_opps"].to_numpy())
    return output


def _sim_rank_table(
    players: pl.DataFrame,
    team_opps: pl.DataFrame,
    stat: str,
    team_col: str,
    participation_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    mean_col = f"{stat}_mean"
    p90_col = f"{stat}_p90"
    denominator = f"team_{stat}_mean"
    team_denoms = team_opps.select(
        "game", "team_id", pl.col(team_col).alias(denominator)
    )
    base = (
        players.filter(pl.col(mean_col) > 0.001)
        .join(team_denoms, on=["game", "team_id"], how="left")
        .sort(["game", "team_id", mean_col], descending=[False, False, True])
        .with_columns(
            pl.col(mean_col).rank("ordinal", descending=True).over(["game", "team_id"]).alias("rank"),
            (pl.col(mean_col) / pl.col(denominator).clip(lower_bound=0.01)).alias("share"),
        )
    )
    breadth = (
        players.group_by(["game", "team_id"])
        .agg(pl.col(participation_col).sum().alias("expected_players_with_opps"))
    )
    ranked = base.join(breadth, on=["game", "team_id"], how="left").select(
        "game", "team_id", "player", "position", "rank", mean_col, p90_col,
        "share", "expected_players_with_opps",
    )
    return ranked, breadth


def _sim_rank_metrics(ranked: pl.DataFrame, prefix: str, stat: str) -> dict[str, dict[str, float]]:
    mean_col = f"{stat}_mean"
    p90_col = f"{stat}_p90"
    output: dict[str, dict[str, float]] = {}
    for rank in (1, 2, 3):
        sub = ranked.filter(pl.col("rank") == rank)
        output[f"{prefix}_rank{rank}_mean_count"] = _summary(sub[mean_col].to_numpy())
        output[f"{prefix}_rank{rank}_p90_count"] = _summary(sub[p90_col].to_numpy())
        output[f"{prefix}_rank{rank}_share"] = _summary(sub["share"].to_numpy())
    leaders = ranked.filter(pl.col("rank") == 1)
    output[f"{prefix}_expected_players_with_opps"] = _summary(
        leaders["expected_players_with_opps"].to_numpy()
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--team-opportunity", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/player-opportunity-realism"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    needed = [
        "game_id", "posteam", "pass_attempt", "rush_attempt",
        "receiver_player_id", "rusher_player_id", "passer_player_id",
        "qb_kneel", "qb_spike",
    ]
    pbp = pbp.select([c for c in needed if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    target_events = pbp.with_columns(
        ((pl.col("pass_attempt").fill_null(0) == 1) & pl.col("receiver_player_id").is_not_null())
        .cast(pl.Int16).alias("target_event")
    )
    rush_events = pbp.with_columns(
        ((pl.col("rush_attempt").fill_null(0) == 1) & pl.col("rusher_player_id").is_not_null())
        .cast(pl.Int16).alias("rush_event")
    )
    hist_targets = _ranked_team_game(target_events, "target_event", "receiver_player_id")
    hist_rushes = _ranked_team_game(rush_events, "rush_event", "rusher_player_id")

    qb_games = (
        pbp.filter((pl.col("pass_attempt").fill_null(0) == 1) & pl.col("passer_player_id").is_not_null())
        .group_by(["game_id", "posteam", "passer_player_id"])
        .agg(pl.len().alias("pass_attempts"))
        .sort(["game_id", "posteam", "pass_attempts"], descending=[False, False, True])
        .with_columns(pl.col("pass_attempts").rank("ordinal", descending=True).over(["game_id", "posteam"]).alias("rank"))
    )
    hist_qb1 = qb_games.filter(pl.col("rank") == 1)

    players = _read(args.players)
    team_opps = _read(args.team_opportunity)
    sim_targets, target_breadth = _sim_rank_table(
        players, team_opps, "targets", "targets_mean", "target_participation_probability"
    )
    sim_rushes, rush_breadth = _sim_rank_table(
        players, team_opps, "rush_attempts", "rush_attempts_mean", "rush_participation_probability"
    )
    sim_qbs = (
        players.filter((pl.col("position") == "QB") & (pl.col("pass_attempts_mean") > 0.1))
        .sort(["game", "team_id", "pass_attempts_mean"], descending=[False, False, True])
        .with_columns(pl.col("pass_attempts_mean").rank("ordinal", descending=True).over(["game", "team_id"]).alias("rank"))
        .filter(pl.col("rank") == 1)
    )

    historical = {
        **_historical_rank_metrics(hist_targets, "target"),
        **_historical_rank_metrics(hist_rushes, "rush"),
        "qb1_pass_attempts": _summary(hist_qb1["pass_attempts"].to_numpy()),
    }
    simulated = {
        **_sim_rank_metrics(sim_targets, "target", "targets"),
        **_sim_rank_metrics(sim_rushes, "rush", "rush_attempts"),
        "qb1_pass_attempts_mean": _summary(sim_qbs["pass_attempts_mean"].to_numpy()),
        "qb1_pass_attempts_p90": _summary(sim_qbs["pass_attempts_p90"].to_numpy()),
    }

    comparisons = []
    for family, hist_ranked, sim_ranked, breadth in (
        ("target", hist_targets, sim_targets, target_breadth),
        ("rush", hist_rushes, sim_rushes, rush_breadth),
    ):
        hist_leader_share = float(hist_ranked.filter(pl.col("rank") == 1)["share"].mean())
        sim_leader_share = float(sim_ranked.filter(pl.col("rank") == 1)["share"].mean())
        hist_second_share = float(hist_ranked.filter(pl.col("rank") == 2)["share"].mean())
        sim_second_share = float(sim_ranked.filter(pl.col("rank") == 2)["share"].mean())
        hist_breadth = float(hist_ranked.filter(pl.col("rank") == 1)["players_with_opps"].mean())
        sim_breadth = float(breadth["expected_players_with_opps"].mean())
        comparisons.append({
            "family": family,
            "historical_leader_share": hist_leader_share,
            "simulated_leader_share": sim_leader_share,
            "leader_share_delta": sim_leader_share - hist_leader_share,
            "historical_second_share": hist_second_share,
            "simulated_second_share": sim_second_share,
            "second_share_delta": sim_second_share - hist_second_share,
            "historical_players_with_opps": hist_breadth,
            "simulated_expected_players_with_opps": sim_breadth,
            "breadth_delta": sim_breadth - hist_breadth,
        })
    comparison = pl.DataFrame(comparisons)

    max_leader_delta = float(comparison["leader_share_delta"].abs().max())
    max_breadth_delta = float(comparison["breadth_delta"].abs().max())
    if max_leader_delta > 0.08:
        diagnosis = "role_concentration_mismatch"
    elif max_breadth_delta > 1.5:
        diagnosis = "opportunity_breadth_mismatch"
    else:
        diagnosis = "role_concentration_within_broad_historical_band"

    args.out.mkdir(parents=True, exist_ok=True)
    hist_targets.write_csv(args.out / "historical_target_ranked_games.csv")
    hist_rushes.write_csv(args.out / "historical_rush_ranked_games.csv")
    sim_targets.write_csv(args.out / "simulated_target_ranked.csv")
    sim_rushes.write_csv(args.out / "simulated_rush_ranked.csv")
    comparison.write_csv(args.out / "concentration_comparison.csv")

    manifest = {
        "artifact": "Monster Player Opportunity Realism Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "historical_team_games": int(hist_targets.select(["game_id", "posteam"]).unique().height),
        "simulated_team_games": int(team_opps.height),
        "historical": historical,
        "simulated": simulated,
        "concentration_comparison": comparison.to_dicts(),
        "diagnosis": diagnosis,
        "market_blind": True,
        "principle": "Validate role hierarchy and expected game participation after finite football supply is realistic; do not compare nonzero projection means with observed one-game participation.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
