from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

FAMILIES = ("designed_non_sneak", "scramble", "sneak", "kneel")


def _flag(name: str, columns: set[str]) -> pl.Expr:
    if name not in columns:
        return pl.lit(0.0)
    return pl.col(name).fill_null(0).cast(pl.Float64)


def _corr(frame: pl.DataFrame, left: str, right: str) -> float | None:
    if frame.height < 8:
        return None
    a = frame.get_column(left).to_numpy()
    b = frame.get_column(right).to_numpy()
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2022,2023,2024,2025")
    parser.add_argument("--out", type=Path, default=Path("artifacts/v7-qb-rush-families"))
    parser.add_argument("--min-starts", type=int, default=4)
    args = parser.parse_args()
    seasons = [int(value) for value in args.seasons.split(",")]

    pbp = nfl.load_pbp(seasons)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    required = {
        "season",
        "game_id",
        "posteam",
        "pass_attempt",
        "rush_attempt",
        "passer_player_id",
        "rusher_player_id",
        "rushing_yards",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    columns = set(pbp.columns)
    lead_passers = (
        pbp.filter(
            (pl.col("pass_attempt").fill_null(0).cast(pl.Float64) == 1.0)
            & pl.col("passer_player_id").is_not_null()
        )
        .group_by(["season", "game_id", "posteam", "passer_player_id"])
        .agg(pl.len().alias("pass_attempts"))
        .sort(
            ["season", "game_id", "posteam", "pass_attempts"],
            descending=[False, False, False, True],
        )
        .group_by(["season", "game_id", "posteam"], maintain_order=True)
        .first()
        .filter(pl.col("pass_attempts") >= 15)
        .rename({"passer_player_id": "qb_id"})
    )

    qb_rush_plays = (
        pbp.filter(
            (pl.col("rush_attempt").fill_null(0).cast(pl.Float64) == 1.0)
            & pl.col("rusher_player_id").is_not_null()
            & pl.col("posteam").is_not_null()
        )
        .join(
            lead_passers.select(["season", "game_id", "posteam", "qb_id"]),
            left_on=["season", "game_id", "posteam", "rusher_player_id"],
            right_on=["season", "game_id", "posteam", "qb_id"],
            how="inner",
        )
        .with_columns(
            pl.when(_flag("qb_kneel", columns) == 1.0)
            .then(pl.lit("kneel"))
            .when(_flag("qb_sneak", columns) == 1.0)
            .then(pl.lit("sneak"))
            .when(_flag("qb_scramble", columns) == 1.0)
            .then(pl.lit("scramble"))
            .otherwise(pl.lit("designed_non_sneak"))
            .alias("rush_family")
        )
    )

    family_games = (
        qb_rush_plays.group_by(
            ["season", "game_id", "posteam", "rusher_player_id", "rush_family"]
        )
        .agg(
            pl.len().alias("attempts"),
            pl.col("rushing_yards").fill_null(0.0).sum().alias("yards"),
        )
        .pivot(
            on="rush_family",
            index=["season", "game_id", "posteam", "rusher_player_id"],
            values=["attempts", "yards"],
            aggregate_function="first",
        )
    )

    # Polars pivot names are value_family. Normalize every expected family so zero
    # events remain explicit rather than disappearing from the starter population.
    rename_map: dict[str, str] = {}
    for family in FAMILIES:
        for value in ("attempts", "yards"):
            for candidate in (f"{value}_{family}", f"{family}_{value}"):
                if candidate in family_games.columns:
                    rename_map[candidate] = f"{family}_{value}"
    if rename_map:
        family_games = family_games.rename(rename_map)

    games = lead_passers.rename({"qb_id": "rusher_player_id"}).join(
        family_games,
        on=["season", "game_id", "posteam", "rusher_player_id"],
        how="left",
    )
    fill_exprs = []
    for family in FAMILIES:
        for value in ("attempts", "yards"):
            name = f"{family}_{value}"
            if name not in games.columns:
                games = games.with_columns(pl.lit(0.0).alias(name))
            fill_exprs.append(pl.col(name).fill_null(0.0).alias(name))
    games = games.with_columns(*fill_exprs).with_columns(
        (
            pl.col("designed_non_sneak_attempts")
            + pl.col("scramble_attempts")
            + pl.col("sneak_attempts")
        ).alias("competitive_qb_rush_attempts"),
        (
            pl.col("designed_non_sneak_yards")
            + pl.col("scramble_yards")
            + pl.col("sneak_yards")
        ).alias("competitive_qb_rush_yards"),
    )

    season_qb = (
        games.group_by(["season", "rusher_player_id"])
        .agg(
            pl.len().alias("starts"),
            *[
                pl.col(f"{family}_attempts").sum().alias(f"{family}_attempts")
                for family in FAMILIES
            ],
            *[
                pl.col(f"{family}_yards").sum().alias(f"{family}_yards")
                for family in FAMILIES
            ],
            pl.col("competitive_qb_rush_attempts").sum().alias(
                "competitive_qb_rush_attempts"
            ),
            pl.col("competitive_qb_rush_yards").sum().alias(
                "competitive_qb_rush_yards"
            ),
        )
        .filter(pl.col("starts") >= args.min_starts)
        .with_columns(
            *[
                (
                    pl.col(f"{family}_attempts")
                    / pl.col("starts").clip(lower_bound=1)
                ).alias(f"{family}_attempts_per_start")
                for family in FAMILIES
            ],
            (
                pl.col("competitive_qb_rush_attempts")
                / pl.col("starts").clip(lower_bound=1)
            ).alias("competitive_attempts_per_start"),
        )
        .sort(["season", "competitive_attempts_per_start"], descending=[False, True])
    )

    next_season = season_qb.select(
        [
            (pl.col("season") - 1).alias("season"),
            pl.col("rusher_player_id"),
            *[
                pl.col(f"{family}_attempts_per_start").alias(
                    f"next_{family}_attempts_per_start"
                )
                for family in FAMILIES
            ],
            pl.col("competitive_attempts_per_start").alias(
                "next_competitive_attempts_per_start"
            ),
        ]
    )
    recurrence = season_qb.join(
        next_season,
        on=["season", "rusher_player_id"],
        how="inner",
    )

    family_summary: dict[str, dict[str, float | None]] = {}
    for family in FAMILIES:
        game_values = games.get_column(f"{family}_attempts").to_numpy()
        season_values = season_qb.get_column(f"{family}_attempts_per_start").to_numpy()
        family_summary[family] = {
            "mean_attempts_per_start_game_population": float(np.mean(game_values)),
            "p50_attempts_per_start_game_population": float(np.median(game_values)),
            "p90_attempts_per_start_game_population": float(np.quantile(game_values, 0.90)),
            "mean_season_rate_per_start": float(np.mean(season_values)),
            "next_season_recurrence_corr": _corr(
                recurrence,
                f"{family}_attempts_per_start",
                f"next_{family}_attempts_per_start",
            ),
        }

    manifest = {
        "seasons": seasons,
        "lead_passer_game_rows": int(games.height),
        "qb_season_rows": int(season_qb.height),
        "next_season_pairs": int(recurrence.height),
        "minimum_starts_per_qb_season": args.min_starts,
        "historical_family_source": {
            "kneel": "nflverse qb_kneel when present",
            "sneak": "nflverse qb_sneak when present",
            "scramble": "nflverse qb_scramble when present",
            "designed_non_sneak": "remaining lead-QB rush attempts",
        },
        "pressure_vs_coverage_scramble_calibrated_here": False,
        "family_summary": family_summary,
        "competitive_qb_rushing": {
            "mean_attempts_per_start": float(
                games.get_column("competitive_qb_rush_attempts").mean()
            ),
            "mean_yards_per_start": float(
                games.get_column("competitive_qb_rush_yards").mean()
            ),
            "next_season_recurrence_corr": _corr(
                recurrence,
                "competitive_attempts_per_start",
                "next_competitive_attempts_per_start",
            ),
        },
        "market_blind": True,
        "week1_2026_used": False,
        "principle": (
            "QB rushing is not one reservoir. Separate designed keeps, sneaks, "
            "scrambles and kneels before assigning player tendency or game-level variance."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "qb_rush_family_games.csv")
    season_qb.write_csv(args.out / "qb_rush_family_seasons.csv")
    recurrence.write_csv(args.out / "qb_rush_family_recurrence.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
