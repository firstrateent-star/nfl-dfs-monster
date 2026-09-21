from __future__ import annotations

import polars as pl


def build_reality_benchmark(
    projected: pl.DataFrame,
    actual: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float | int]]:
    required_projected = {
        "game",
        "away",
        "home",
        "projected_winner",
        "away_points_mean",
        "home_points_mean",
        "away_points_p10",
        "away_points_p90",
        "home_points_p10",
        "home_points_p90",
        "total_mean",
        "margin_mean",
    }
    required_actual = {
        "game",
        "away",
        "home",
        "actual_away_points",
        "actual_home_points",
    }
    missing_projected = sorted(required_projected.difference(projected.columns))
    missing_actual = sorted(required_actual.difference(actual.columns))
    if missing_projected:
        raise ValueError(f"projected data missing columns: {missing_projected}")
    if missing_actual:
        raise ValueError(f"actual data missing columns: {missing_actual}")

    joined = projected.join(actual, on=["game", "away", "home"], how="inner")
    if joined.height != actual.height:
        raise ValueError(
            f"expected {actual.height} actual games, matched {joined.height}"
        )

    detail = joined.with_columns(
        (pl.col("actual_away_points") + pl.col("actual_home_points")).alias(
            "actual_total"
        ),
        (pl.col("actual_away_points") - pl.col("actual_home_points")).alias(
            "actual_margin"
        ),
        pl.when(pl.col("actual_away_points") > pl.col("actual_home_points"))
        .then(pl.col("away"))
        .when(pl.col("actual_home_points") > pl.col("actual_away_points"))
        .then(pl.col("home"))
        .otherwise(pl.lit("TIE"))
        .alias("actual_winner"),
    ).with_columns(
        (pl.col("projected_winner") == pl.col("actual_winner")).alias(
            "winner_correct"
        ),
        (pl.col("away_points_mean") - pl.col("actual_away_points"))
        .abs()
        .alias("away_score_abs_error"),
        (pl.col("home_points_mean") - pl.col("actual_home_points"))
        .abs()
        .alias("home_score_abs_error"),
        (pl.col("total_mean") - pl.col("actual_total"))
        .abs()
        .alias("total_abs_error"),
        (pl.col("margin_mean") - pl.col("actual_margin"))
        .abs()
        .alias("margin_abs_error"),
        (
            (pl.col("actual_away_points") >= pl.col("away_points_p10"))
            & (pl.col("actual_away_points") <= pl.col("away_points_p90"))
        ).alias("away_inside_p10_p90"),
        (
            (pl.col("actual_home_points") >= pl.col("home_points_p10"))
            & (pl.col("actual_home_points") <= pl.col("home_points_p90"))
        ).alias("home_inside_p10_p90"),
        (pl.col("actual_away_points") < pl.col("away_points_p10")).alias(
            "away_below_p10"
        ),
        (pl.col("actual_home_points") < pl.col("home_points_p10")).alias(
            "home_below_p10"
        ),
        (pl.col("actual_away_points") > pl.col("away_points_p90")).alias(
            "away_above_p90"
        ),
        (pl.col("actual_home_points") > pl.col("home_points_p90")).alias(
            "home_above_p90"
        ),
    )

    team_abs_errors = pl.concat(
        [
            detail.select(pl.col("away_score_abs_error").alias("error")),
            detail.select(pl.col("home_score_abs_error").alias("error")),
        ]
    )
    inside = pl.concat(
        [
            detail.select(pl.col("away_inside_p10_p90").alias("inside")),
            detail.select(pl.col("home_inside_p10_p90").alias("inside")),
        ]
    )
    below = int(
        detail.select(
            pl.col("away_below_p10").sum() + pl.col("home_below_p10").sum()
        ).item()
    )
    above = int(
        detail.select(
            pl.col("away_above_p90").sum() + pl.col("home_above_p90").sum()
        ).item()
    )
    projected_total_mean = float(detail.get_column("total_mean").mean())
    actual_total_mean = float(detail.get_column("actual_total").mean())

    summary: dict[str, float | int] = {
        "games": detail.height,
        "winner_accuracy": float(detail.get_column("winner_correct").mean()),
        "team_score_mae": float(team_abs_errors.get_column("error").mean()),
        "game_total_mae": float(detail.get_column("total_abs_error").mean()),
        "margin_mae": float(detail.get_column("margin_abs_error").mean()),
        "team_score_p10_p90_coverage": float(inside.get_column("inside").mean()),
        "team_scores_below_p10": below,
        "team_scores_above_p90": above,
        "projected_game_total_mean": projected_total_mean,
        "actual_game_total_mean": actual_total_mean,
        "game_total_bias": projected_total_mean - actual_total_mean,
    }
    return detail.sort("total_abs_error", descending=True), summary
