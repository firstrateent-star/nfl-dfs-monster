from __future__ import annotations

import polars as pl


REGIMES = ("collapse", "fragile", "normal", "surge")
_KEYS = ("game", "player_id", "player", "position", "team")


def _require_columns(frame: pl.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _with_opponent(frame: pl.DataFrame) -> pl.DataFrame:
    split = pl.col("game").str.split_exact("@", 1)
    return (
        frame.with_columns(
            split.struct.field("field_0").alias("_away"),
            split.struct.field("field_1").alias("_home"),
        )
        .with_columns(
            pl.when(pl.col("team") == pl.col("_away"))
            .then(pl.col("_home"))
            .otherwise(pl.col("_away"))
            .alias("opponent")
        )
        .drop("_away", "_home")
    )


def build_premise_robustness(
    player_world: pl.DataFrame,
    premise: pl.DataFrame,
    *,
    minimum_regime_worlds: int = 3,
) -> pl.DataFrame:
    _require_columns(
        player_world,
        {
            "game",
            "world",
            "player_id",
            "player",
            "position",
            "team",
            "fanduel_points",
            "pass_attempts",
            "targets",
            "rush_attempts",
        },
        name="player-world telemetry",
    )
    _require_columns(
        premise,
        {
            "game",
            "world",
            "team",
            "regime",
            "offensive_cohesion",
            "defensive_regime",
            "defensive_cohesion",
        },
        name="world-premise telemetry",
    )

    worlds = _with_opponent(player_world)
    own = premise.select(
        "game",
        "world",
        "team",
        pl.col("regime").alias("offensive_regime"),
        "offensive_cohesion",
    )
    opponent = premise.select(
        "game",
        "world",
        pl.col("team").alias("opponent"),
        pl.col("defensive_regime").alias("opponent_defensive_regime"),
        pl.col("defensive_cohesion").alias("opponent_defensive_cohesion"),
    )
    joined = worlds.join(own, on=["game", "world", "team"], how="left").join(
        opponent,
        on=["game", "world", "opponent"],
        how="left",
    )

    if joined.get_column("offensive_regime").null_count() > 0:
        raise ValueError("player worlds could not all be joined to their offensive premise")
    if joined.get_column("opponent_defensive_regime").null_count() > 0:
        raise ValueError("player worlds could not all be joined to opponent defensive premise")

    opportunity = (
        pl.col("pass_attempts").fill_null(0.0)
        + pl.col("targets").fill_null(0.0)
        + pl.col("rush_attempts").fill_null(0.0)
    )
    result = joined.group_by(list(_KEYS)).agg(
        pl.len().alias("worlds"),
        pl.col("fanduel_points").mean().alias("fd_mean"),
        pl.col("fanduel_points").quantile(0.50).alias("fd_p50"),
        pl.col("fanduel_points").quantile(0.95).alias("fd_p95"),
        pl.col("fanduel_points").quantile(0.99).alias("fd_p99"),
        (pl.col("fanduel_points") >= 15.0).mean().alias("fd_15_plus_probability"),
        (pl.col("fanduel_points") >= 20.0).mean().alias("fd_20_plus_probability"),
        (pl.col("fanduel_points") >= 25.0).mean().alias("fd_25_plus_probability"),
        (opportunity <= 0.0).mean().alias("zero_opportunity_probability"),
        pl.col("offensive_cohesion").mean().alias("mean_offensive_cohesion"),
        pl.col("opponent_defensive_cohesion")
        .mean()
        .alias("mean_opponent_defensive_cohesion"),
    )

    valid_offense_means: list[str] = []
    valid_offense_probs: list[str] = []
    for regime in REGIMES:
        stats = (
            joined.filter(pl.col("offensive_regime") == regime)
            .group_by(list(_KEYS))
            .agg(
                pl.len().alias(f"{regime}_worlds"),
                pl.col("fanduel_points").mean().alias(f"{regime}_fd_mean"),
                pl.col("fanduel_points").quantile(0.90).alias(f"{regime}_fd_p90"),
                (pl.col("fanduel_points") >= 20.0)
                .mean()
                .alias(f"{regime}_fd_20_plus_probability"),
            )
        )
        result = result.join(stats, on=list(_KEYS), how="left")
        valid_mean = f"_{regime}_valid_mean"
        valid_prob = f"_{regime}_valid_prob"
        valid_offense_means.append(valid_mean)
        valid_offense_probs.append(valid_prob)
        result = result.with_columns(
            pl.when(pl.col(f"{regime}_worlds").fill_null(0) >= minimum_regime_worlds)
            .then(pl.col(f"{regime}_fd_mean"))
            .otherwise(None)
            .alias(valid_mean),
            pl.when(pl.col(f"{regime}_worlds").fill_null(0) >= minimum_regime_worlds)
            .then(pl.col(f"{regime}_fd_20_plus_probability"))
            .otherwise(None)
            .alias(valid_prob),
        )

    result = result.with_columns(
        pl.min_horizontal(*valid_offense_means).alias("worst_regime_fd_mean"),
        (
            pl.max_horizontal(*valid_offense_means)
            - pl.min_horizontal(*valid_offense_means)
        ).alias("regime_fd_mean_span"),
        pl.min_horizontal(*valid_offense_probs).alias(
            "worst_regime_fd_20_plus_probability"
        ),
        pl.when(pl.col("_normal_valid_mean") > 0.0)
        .then(pl.col("_collapse_valid_mean") / pl.col("_normal_valid_mean"))
        .otherwise(None)
        .alias("collapse_retention_vs_normal"),
        pl.when(pl.col("_normal_valid_mean") > 0.0)
        .then(pl.col("_fragile_valid_mean") / pl.col("_normal_valid_mean"))
        .otherwise(None)
        .alias("fragile_retention_vs_normal"),
    )

    for regime in REGIMES:
        stats = (
            joined.filter(pl.col("opponent_defensive_regime") == regime)
            .group_by(list(_KEYS))
            .agg(
                pl.len().alias(f"opp_def_{regime}_worlds"),
                pl.col("fanduel_points").mean().alias(f"opp_def_{regime}_fd_mean"),
            )
        )
        result = result.join(stats, on=list(_KEYS), how="left")

    return (
        result.drop(valid_offense_means + valid_offense_probs)
        .with_columns(
            pl.when(pl.col("fd_mean") > 0.0)
            .then(pl.col("regime_fd_mean_span") / pl.col("fd_mean"))
            .otherwise(None)
            .alias("premise_dependency_ratio")
        )
        .sort("fd_p95", descending=True)
    )
