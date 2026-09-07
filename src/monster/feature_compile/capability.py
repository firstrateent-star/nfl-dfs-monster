from __future__ import annotations

import polars as pl

_COMBINE_COLUMNS = [
    "pfr_id",
    "forty",
    "bench",
    "vertical",
    "broad_jump",
    "cone",
    "shuttle",
]

_DEFENSE_SUM_COLUMNS = [
    "def_times_blitzed",
    "def_times_hurried",
    "def_times_hitqb",
    "def_times_pressured",
    "def_sacks",
    "def_qb_hits",
    "def_tackles",
    "def_tackles_for_loss",
    "def_interceptions",
    "def_pass_defended",
]
_DEFENSE_MEAN_COLUMNS = ["def_times_pressured_pct"]


def compile_combine_features(combine: pl.DataFrame) -> pl.DataFrame:
    """Keep raw combine evidence plus a bounded-friendly size/speed derivative.

    Raw measurements remain authoritative evidence; derived scores are mechanism inputs,
    never direct point bonuses. Missing drills stay null rather than being treated as zero.
    """
    if "pfr_id" not in combine.columns:
        return pl.DataFrame()
    selected = [c for c in _COMBINE_COLUMNS if c in combine.columns]
    frame = combine.select(selected).with_columns(pl.col("pfr_id").cast(pl.Utf8))
    if "forty" in frame.columns and "wt" in combine.columns:
        weight = combine.select(["pfr_id", "wt"]).with_columns(pl.col("pfr_id").cast(pl.Utf8))
        frame = frame.join(weight, on="pfr_id", how="left").with_columns(
            pl.when((pl.col("forty") > 0) & pl.col("wt").is_not_null())
            .then((pl.col("wt") * 200.0) / (pl.col("forty") ** 4))
            .otherwise(None)
            .alias("combine_speed_score")
        )
    return frame.unique(subset=["pfr_id"], keep="last")


def compile_defender_history(pfr_defense_weekly: pl.DataFrame) -> pl.DataFrame:
    """Aggregate recent PFR defensive evidence without collapsing mechanisms together."""
    if not pfr_defense_weekly.height or "pfr_player_id" not in pfr_defense_weekly.columns:
        return pl.DataFrame()

    sum_cols = [c for c in _DEFENSE_SUM_COLUMNS if c in pfr_defense_weekly.columns]
    mean_cols = [c for c in _DEFENSE_MEAN_COLUMNS if c in pfr_defense_weekly.columns]
    if not sum_cols and not mean_cols:
        return pfr_defense_weekly.select("pfr_player_id").unique().with_columns(
            pl.lit(0, dtype=pl.Int64).alias("defense_games_observed")
        )

    aggs: list[pl.Expr] = [pl.len().alias("defense_games_observed")]
    aggs.extend(pl.col(c).sum().alias(f"hist_{c}") for c in sum_cols)
    aggs.extend(pl.col(c).mean().alias(f"hist_{c}") for c in mean_cols)
    return (
        pfr_defense_weekly.with_columns(pl.col("pfr_player_id").cast(pl.Utf8))
        .group_by("pfr_player_id")
        .agg(*aggs)
    )


def attach_capability_evidence(
    personnel: pl.DataFrame,
    combine: pl.DataFrame,
    pfr_defense_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Attach free physical and defensive evidence to the full league personnel table."""
    result = personnel
    combine_features = compile_combine_features(combine)
    if combine_features.height and "pfr_id" in result.columns:
        result = result.with_columns(pl.col("pfr_id").cast(pl.Utf8)).join(
            combine_features,
            on="pfr_id",
            how="left",
            suffix="_combine",
        )

    defense = compile_defender_history(pfr_defense_weekly)
    if defense.height and "pfr_id" in result.columns:
        result = result.join(
            defense,
            left_on="pfr_id",
            right_on="pfr_player_id",
            how="left",
        ).with_columns(pl.col("defense_games_observed").fill_null(0))
    elif "defense_games_observed" not in result.columns:
        result = result.with_columns(pl.lit(0, dtype=pl.Int64).alias("defense_games_observed"))
    return result


def capability_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.len().alias("roster_rows"),
        (pl.col("defense_games_observed") > 0).sum().alias("players_with_defender_history"),
    ]
    for column, alias in [
        ("forty", "players_with_forty"),
        ("bench", "players_with_bench"),
        ("vertical", "players_with_vertical"),
        ("broad_jump", "players_with_broad_jump"),
        ("cone", "players_with_cone"),
        ("shuttle", "players_with_shuttle"),
        ("combine_speed_score", "players_with_speed_score"),
    ]:
        if column in snapshot.columns:
            expressions.append(pl.col(column).is_not_null().sum().alias(alias))
    return snapshot.group_by("team_id").agg(*expressions).sort("team_id")
