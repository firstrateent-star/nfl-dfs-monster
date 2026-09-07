from __future__ import annotations

import polars as pl

from monster.teams import NFL_TEAMS, TEAM_ALIASES

_TEAM_MAP = {**{team: team for team in NFL_TEAMS}, **TEAM_ALIASES}


def compile_latest_depth_chart(depth_charts: pl.DataFrame) -> pl.DataFrame:
    """Reduce the 2025+ dated ESPN depth-chart feed to one current role per team/player.

    Players can appear in more than one team's historical daily charts after trades,
    waivers or camp moves. Team identity is therefore part of the depth-chart key.
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
        pl.col("team")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .replace(_TEAM_MAP)
        .alias("depth_team_id"),
        pl.col("pos_rank").cast(pl.Int64, strict=False),
    ).filter(
        pl.col("gsis_id").is_not_null()
        & pl.col("depth_team_id").is_in(list(NFL_TEAMS))
    )

    return (
        frame.sort(["dt", "pos_rank"], descending=[True, False])
        .unique(subset=["depth_team_id", "gsis_id"], keep="first")
        .select(
            "depth_team_id",
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
    required = {"gsis_id", "team_id"}
    missing = required.difference(personnel.columns)
    if missing:
        raise ValueError(f"Personnel snapshot missing depth join keys: {sorted(missing)}")

    before = personnel.height
    joined = personnel.join(
        depth,
        left_on=["team_id", "gsis_id"],
        right_on=["depth_team_id", "gsis_id"],
        how="left",
    )
    if joined.height != before:
        raise ValueError(
            f"Depth-chart join changed personnel row count: before={before} after={joined.height}"
        )
    return joined


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
