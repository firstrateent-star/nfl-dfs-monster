from __future__ import annotations

import polars as pl

from monster.teams import TEAM_ALIASES

SCRIMMAGE_TYPES = ["pass", "run"]


def _require(frame: pl.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")


def _safe_mean(expr: pl.Expr, alias: str) -> pl.Expr:
    return expr.cast(pl.Float64).mean().fill_null(0.0).alias(alias)


def compile_team_policy(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile market-blind team behavior and efficiency priors from play-by-play.

    Expensive historical evaluation happens once here; Monte Carlo consumes a compact team state.
    Team identifiers are normalized before aggregation, and touchdown-drive labels represent
    offensive scoring only rather than generic touchdown events such as pick-sixes.
    """
    required = {
        "game_id",
        "posteam",
        "defteam",
        "play_type",
        "down",
        "yardline_100",
        "game_seconds_remaining",
        "score_differential",
        "epa",
        "success",
        "touchdown",
        "pass_touchdown",
        "rush_touchdown",
        "interception",
        "fumble_lost",
        "sack",
        "qb_hit",
        "qb_dropback",
        "field_goal_attempt",
        "field_goal_result",
        "fixed_drive",
    }
    _require(pbp, required)

    pbp = pbp.with_columns(
        pl.col("posteam").replace(TEAM_ALIASES),
        pl.col("defteam").replace(TEAM_ALIASES),
    )
    offensive_touchdown = (
        (pl.col("pass_touchdown").fill_null(0) == 1)
        | (pl.col("rush_touchdown").fill_null(0) == 1)
    )

    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null() & pl.col("play_type").is_in(SCRIMMAGE_TYPES)
    )
    neutral = scrimmage.filter(
        (pl.col("score_differential").abs() <= 7) & (pl.col("game_seconds_remaining") > 900)
    )

    pace = (
        neutral.sort(
            ["game_id", "fixed_drive", "game_seconds_remaining"],
            descending=[False, False, True],
        )
        .with_columns(
            -pl.col("game_seconds_remaining")
            .diff()
            .over(["game_id", "fixed_drive"])
            .alias("seconds_between_plays")
        )
        .filter(pl.col("seconds_between_plays").is_between(5, 60))
        .group_by("posteam")
        .agg(pl.col("seconds_between_plays").mean().alias("neutral_seconds_per_play"))
        .rename({"posteam": "team_id"})
    )

    explosive_expr = (
        (pl.col("yards_gained") >= 15).cast(pl.Float64)
        if "yards_gained" in scrimmage.columns
        else pl.lit(0.0)
    )
    offense = (
        scrimmage.with_columns(
            explosive_expr.alias("is_explosive"),
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
            .cast(pl.Float64)
            .alias("is_turnover"),
        )
        .group_by("posteam")
        .agg(
            pl.len().alias("scrimmage_plays"),
            pl.col("game_id").n_unique().alias("games_observed"),
            pl.col("epa").mean().alias("offensive_epa_per_play"),
            pl.col("success").mean().alias("offensive_success_rate"),
            pl.col("is_explosive").mean().alias("offensive_explosive_rate"),
            pl.col("is_turnover").mean().alias("turnover_play_rate"),
            (pl.col("sack").sum() / pl.col("qb_dropback").sum().clip(lower_bound=1)).alias(
                "sack_rate_allowed"
            ),
            (pl.col("qb_hit").sum() / pl.col("qb_dropback").sum().clip(lower_bound=1)).alias(
                "qb_hit_rate_allowed"
            ),
            pl.col("pass_touchdown").sum().alias("pass_touchdowns"),
            pl.col("rush_touchdown").sum().alias("rush_touchdowns"),
        )
        .with_columns(
            (pl.col("scrimmage_plays") / pl.col("games_observed").clip(lower_bound=1)).alias(
                "plays_per_game"
            ),
            (
                pl.col("pass_touchdowns")
                / (pl.col("pass_touchdowns") + pl.col("rush_touchdowns")).clip(lower_bound=1)
            ).alias("pass_td_share"),
        )
        .rename({"posteam": "team_id"})
    )

    neutral_policy = (
        neutral.with_columns((pl.col("play_type") == "pass").cast(pl.Float64).alias("is_pass"))
        .group_by("posteam")
        .agg(
            pl.len().alias("neutral_plays"),
            pl.col("is_pass").mean().alias("neutral_pass_rate"),
            pl.col("is_pass")
            .filter(pl.col("down").is_in([1, 2]))
            .mean()
            .alias("early_down_pass_rate"),
        )
        .rename({"posteam": "team_id"})
    )

    drive_level = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("fixed_drive").is_not_null())
        .with_columns(offensive_touchdown.cast(pl.Int8).alias("offensive_touchdown"))
        .group_by(["game_id", "posteam", "fixed_drive"])
        .agg(
            pl.col("offensive_touchdown").max().fill_null(0).alias("drive_td"),
            (pl.col("field_goal_result") == "made").max().fill_null(False).alias("drive_fg"),
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
            .max()
            .fill_null(False)
            .alias("drive_turnover"),
            pl.len().alias("drive_plays"),
        )
    )

    drives = (
        drive_level.group_by("posteam")
        .agg(
            pl.len().alias("drives"),
            pl.col("game_id").n_unique().alias("drive_games"),
            _safe_mean(pl.col("drive_td"), "td_drive_rate"),
            _safe_mean(pl.col("drive_fg"), "fg_drive_rate"),
            _safe_mean(pl.col("drive_turnover"), "turnover_drive_rate"),
            pl.col("drive_plays").mean().alias("plays_per_drive"),
        )
        .with_columns(
            (pl.col("drives") / pl.col("drive_games").clip(lower_bound=1)).alias(
                "drives_per_game"
            )
        )
        .rename({"posteam": "team_id"})
    )

    red_zone = (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & (pl.col("yardline_100") <= 20)
        )
        .with_columns(offensive_touchdown.cast(pl.Int8).alias("offensive_touchdown"))
        .group_by(["game_id", "posteam", "fixed_drive"])
        .agg(pl.col("offensive_touchdown").max().fill_null(0).alias("rz_drive_td"))
        .group_by("posteam")
        .agg(
            pl.len().alias("red_zone_drives"),
            _safe_mean(pl.col("rz_drive_td"), "red_zone_td_rate"),
        )
        .rename({"posteam": "team_id"})
    )

    defense = (
        scrimmage.with_columns(
            explosive_expr.alias("is_explosive"),
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
            .cast(pl.Float64)
            .alias("is_takeaway"),
        )
        .group_by("defteam")
        .agg(
            pl.col("epa").mean().alias("defensive_epa_allowed_per_play"),
            pl.col("success").mean().alias("defensive_success_rate_allowed"),
            pl.col("is_explosive").mean().alias("defensive_explosive_rate_allowed"),
            pl.col("is_takeaway").mean().alias("takeaway_play_rate"),
            (pl.col("sack").sum() / pl.col("qb_dropback").sum().clip(lower_bound=1)).alias(
                "defensive_sack_rate"
            ),
            (pl.col("qb_hit").sum() / pl.col("qb_dropback").sum().clip(lower_bound=1)).alias(
                "defensive_qb_hit_rate"
            ),
        )
        .rename({"defteam": "team_id"})
    )

    drive_defense_map = (
        pbp.select(["game_id", "posteam", "defteam", "fixed_drive"])
        .drop_nulls(["posteam", "defteam", "fixed_drive"])
        .unique(subset=["game_id", "posteam", "fixed_drive"])
        .rename({"posteam": "offense_team"})
    )
    defensive_drive_level = drive_level.rename({"posteam": "offense_team"}).join(
        drive_defense_map,
        on=["game_id", "offense_team", "fixed_drive"],
        how="left",
    )
    defensive_drives = (
        defensive_drive_level.group_by("defteam")
        .agg(
            _safe_mean(pl.col("drive_td"), "defensive_td_drive_rate_allowed"),
            _safe_mean(pl.col("drive_fg"), "defensive_fg_drive_rate_allowed"),
            _safe_mean(pl.col("drive_turnover"), "defensive_takeaway_drive_rate"),
        )
        .rename({"defteam": "team_id"})
    )

    return (
        offense.join(neutral_policy, on="team_id", how="left")
        .join(pace, on="team_id", how="left")
        .join(drives, on="team_id", how="left")
        .join(red_zone, on="team_id", how="left")
        .join(defense, on="team_id", how="left")
        .join(defensive_drives, on="team_id", how="left")
        .sort("team_id")
    )
