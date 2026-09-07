from __future__ import annotations

import polars as pl


def compile_latest_depth_chart(depth_charts: pl.DataFrame) -> pl.DataFrame:
    """Reduce the 2025+ dated ESPN depth-chart feed to one current role per player.

    The modern nflverse schema is dated (`dt`) and uses gsis_id plus positional rank.
    We preserve the raw role fields and never interpret a depth rank as guaranteed snaps.
    """
    required = {"dt", "team", "gsis_id", "pos_grp", "pos_abb", "pos_slot", "pos_rank"}
    missing = required.difference(depth_charts.columns)
    if missing:
        if not depth_charts.height:
            return pl.DataFrame()
        raise ValueError(f"Depth chart missing required modern columns: {sorted(missing)}")

    frame = depth_charts.with_columns(
        pl.col("dt").cast(pl.Datetime, strict=False),
        pl.col("gsis_id").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8).str.to_uppercase(),
        pl.col("pos_rank").cast(pl.Int64, strict=False),
    ).filter(pl.col("gsis_id").is_not_null())

    # A player can appear in more than one slot. Keep his most recent chart entry and,
    # within the same timestamp, the best (lowest) positional rank.
    return (
        frame.sort(["dt", "pos_rank"], descending=[True, False])
        .unique(subset=["team", "gsis_id"], keep="first")
        .select(
            pl.col("team").alias("depth_team_id"),
            "gsis_id",
            pl.col("dt").alias("depth_chart_timestamp"),
            pl.col("pos_grp").alias("depth_position_group"),
            pl.col("pos_abb").alias("depth_position"),
            pl.col("pos_slot").alias("depth_slot"),
            pl.col("pos_rank").alias("depth_rank"),
        )
    )


def attach_depth_chart(personnel: pl.DataFrame, depth_charts: pl.DataFrame) -> pl.DataFrame:
    depth = compile_latest_depth_chart(depth_charts)
    if not depth.height:
        return personnel.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("depth_rank"),
            pl.lit(None, dtype=pl.Utf8).alias("depth_position"),
            pl.lit(None, dtype=pl.Utf8).alias("depth_slot"),
        )
    if "gsis_id" not in personnel.columns:
        raise ValueError("Personnel snapshot requires gsis_id for depth-chart join")
    return personnel.join(depth, on="gsis_id", how="left")


def depth_role_multiplier(rank: pl.Expr) -> pl.Expr:
    """Bounded role multiplier used only to refine uncertain participation priors."""
    return (
        pl.when(rank == 1)
        .then(1.18)
        .when(rank == 2)
        .then(0.88)
        .when(rank == 3)
        .then(0.62)
        .when(rank >= 4)
        .then(0.40)
        .otherwise(1.0)
    )
