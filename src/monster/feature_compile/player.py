from __future__ import annotations

from datetime import date

import polars as pl


def bmi(height_in: float | None, weight_lbs: float | None) -> float | None:
    if not height_in or not weight_lbs or height_in <= 0:
        return None
    return 703.0 * weight_lbs / (height_in * height_in)


def speed_score(weight_lbs: float | None, forty_time: float | None) -> float | None:
    if not weight_lbs or not forty_time or forty_time <= 0:
        return None
    return (weight_lbs * 200.0) / (forty_time**4)


def age_on_date(birth_date: date | None, game_date: date) -> float | None:
    if birth_date is None:
        return None
    return (game_date - birth_date).days / 365.2425


def compile_player_usage(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile player opportunity and efficiency priors directly from football events."""
    required = {
        "posteam",
        "receiver_player_id",
        "rusher_player_id",
        "pass_attempt",
        "rush_attempt",
        "yardline_100",
        "touchdown",
        "yards_gained",
        "air_yards",
        "complete_pass",
        "pass_touchdown",
        "rush_touchdown",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing player-usage columns: {sorted(missing)}")

    targets = (
        pbp.filter((pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null())
        .with_columns(
            (pl.col("yardline_100") <= 20).cast(pl.Int64).alias("red_zone_opportunity"),
            (pl.col("yards_gained") >= 15).cast(pl.Int64).alias("explosive_play"),
        )
        .group_by(["posteam", "receiver_player_id"])
        .agg(
            pl.len().alias("targets"),
            pl.col("complete_pass").sum().alias("receptions"),
            pl.col("air_yards").mean().alias("adot"),
            pl.col("yards_gained").sum().alias("receiving_yards"),
            pl.col("pass_touchdown").sum().alias("receiving_tds"),
            pl.col("red_zone_opportunity").sum().alias("red_zone_targets"),
            pl.col("explosive_play").sum().alias("explosive_receptions"),
        )
        .rename({"receiver_player_id": "player_id", "posteam": "team_id"})
    )
    team_targets = targets.group_by("team_id").agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("red_zone_targets").sum().alias("team_red_zone_targets"),
        pl.col("receiving_tds").sum().alias("team_receiving_tds"),
    )
    targets = targets.join(team_targets, on="team_id", how="left").with_columns(
        (pl.col("targets") / pl.col("team_targets").clip(lower_bound=1)).alias("target_share"),
        (pl.col("red_zone_targets") / pl.col("team_red_zone_targets").clip(lower_bound=1)).alias(
            "red_zone_target_share"
        ),
        (pl.col("receiving_tds") / pl.col("team_receiving_tds").clip(lower_bound=1)).alias(
            "receiving_td_share"
        ),
        (pl.col("receptions") / pl.col("targets").clip(lower_bound=1)).alias("catch_rate"),
        (pl.col("receiving_yards") / pl.col("receptions").clip(lower_bound=1)).alias(
            "yards_per_reception"
        ),
    )

    rushes = (
        pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
        .with_columns(
            (pl.col("yardline_100") <= 20).cast(pl.Int64).alias("red_zone_opportunity"),
            (pl.col("yards_gained") >= 10).cast(pl.Int64).alias("explosive_rush"),
        )
        .group_by(["posteam", "rusher_player_id"])
        .agg(
            pl.len().alias("rushes"),
            pl.col("yards_gained").sum().alias("rushing_yards"),
            pl.col("rush_touchdown").sum().alias("rushing_tds"),
            pl.col("red_zone_opportunity").sum().alias("red_zone_rushes"),
            pl.col("explosive_rush").sum().alias("explosive_rushes"),
        )
        .rename({"rusher_player_id": "player_id", "posteam": "team_id"})
    )
    team_rushes = rushes.group_by("team_id").agg(
        pl.col("rushes").sum().alias("team_rushes"),
        pl.col("red_zone_rushes").sum().alias("team_red_zone_rushes"),
        pl.col("rushing_tds").sum().alias("team_rushing_tds"),
    )
    rushes = rushes.join(team_rushes, on="team_id", how="left").with_columns(
        (pl.col("rushes") / pl.col("team_rushes").clip(lower_bound=1)).alias("rush_share"),
        (pl.col("red_zone_rushes") / pl.col("team_red_zone_rushes").clip(lower_bound=1)).alias(
            "red_zone_rush_share"
        ),
        (pl.col("rushing_tds") / pl.col("team_rushing_tds").clip(lower_bound=1)).alias(
            "rushing_td_share"
        ),
        (pl.col("rushing_yards") / pl.col("rushes").clip(lower_bound=1)).alias(
            "yards_per_carry"
        ),
    )

    return targets.join(rushes, on=["team_id", "player_id"], how="full", coalesce=True).fill_null(0)
