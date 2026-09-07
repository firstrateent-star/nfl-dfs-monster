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
_PLAYER_DEFENSE_COLUMNS = [
    "def_tackles",
    "def_tackles_solo",
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
]


def compile_combine_features(combine: pl.DataFrame) -> pl.DataFrame:
    """Keep raw combine evidence plus a size-adjusted speed derivative."""
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
    """Aggregate PFR pressure evidence without pretending it measures coverage/run defense."""
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


def compile_player_defense_stats(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Aggregate nflverse official defensive stat channels by GSIS player ID."""
    if not player_stats.height or "player_id" not in player_stats.columns:
        return pl.DataFrame()
    fields = [c for c in _PLAYER_DEFENSE_COLUMNS if c in player_stats.columns]
    if not fields:
        return pl.DataFrame()
    return (
        player_stats.with_columns(pl.col("player_id").cast(pl.Utf8))
        .group_by("player_id")
        .agg(
            pl.len().alias("def_stat_games_observed"),
            *[pl.col(c).sum().alias(f"stat_{c}") for c in fields],
        )
    )


def _centered_percentile(metric: pl.Expr, eligible: pl.Expr) -> pl.Expr:
    """Convert a raw rate to roughly [-1,1] relative to positional peers."""
    ranked = metric.rank(method="average").over("position_group")
    count = eligible.cast(pl.Int64).sum().over("position_group").clip(lower_bound=1)
    return pl.when(eligible).then((2.0 * ranked / count - 1.0).clip(-1.0, 1.0)).otherwise(None)


def derive_defender_mechanism_signals(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Create bounded observed pass-rush, coverage and run-defense signals.

    These are starting mechanism priors, not universal defender grades. Each raw channel
    is rate-normalized by observed games and compared only with positional peers.
    """
    games = pl.col("def_stat_games_observed").fill_null(0).cast(pl.Float64).clip(lower_bound=1.0)
    pfr_games = pl.col("defense_games_observed").fill_null(0).cast(pl.Float64).clip(lower_bound=1.0)

    qb_hits = pl.col("stat_def_qb_hits").fill_null(0.0) if "stat_def_qb_hits" in snapshot.columns else pl.lit(0.0)
    sacks = pl.col("stat_def_sacks").fill_null(0.0) if "stat_def_sacks" in snapshot.columns else pl.lit(0.0)
    hurries = (
        pl.col("hist_def_times_hurried").fill_null(0.0)
        if "hist_def_times_hurried" in snapshot.columns
        else pl.lit(0.0)
    )
    pass_defended = (
        pl.col("stat_def_pass_defended").fill_null(0.0)
        if "stat_def_pass_defended" in snapshot.columns
        else pl.lit(0.0)
    )
    interceptions = (
        pl.col("stat_def_interceptions").fill_null(0.0)
        if "stat_def_interceptions" in snapshot.columns
        else pl.lit(0.0)
    )
    tackles = pl.col("stat_def_tackles").fill_null(0.0) if "stat_def_tackles" in snapshot.columns else pl.lit(0.0)
    tfl = (
        pl.col("stat_def_tackles_for_loss").fill_null(0.0)
        if "stat_def_tackles_for_loss" in snapshot.columns
        else pl.lit(0.0)
    )

    frame = snapshot.with_columns(
        ((qb_hits + 1.5 * sacks) / games + 0.5 * hurries / pfr_games).alias("raw_pass_rush_rate"),
        ((pass_defended + 2.0 * interceptions) / games).alias("raw_coverage_play_rate"),
        ((tfl + 0.20 * tackles) / games).alias("raw_run_defense_event_rate"),
    )
    has_stats = pl.col("def_stat_games_observed").fill_null(0) > 0
    group = pl.col("position_group").cast(pl.Utf8)
    return frame.with_columns(
        _centered_percentile(
            pl.col("raw_pass_rush_rate"), has_stats & group.is_in(["DL", "LB"])
        ).alias("observed_pass_rush_signal"),
        _centered_percentile(
            pl.col("raw_coverage_play_rate"), has_stats & group.is_in(["DB", "LB"])
        ).alias("observed_coverage_signal"),
        _centered_percentile(
            pl.col("raw_run_defense_event_rate"), has_stats & group.is_in(["DL", "LB", "DB"])
        ).alias("observed_run_defense_signal"),
    )


def attach_capability_evidence(
    personnel: pl.DataFrame,
    combine: pl.DataFrame,
    pfr_defense_weekly: pl.DataFrame,
    player_stats_history: pl.DataFrame | None = None,
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

    stat_history = compile_player_defense_stats(player_stats_history or pl.DataFrame())
    if stat_history.height and "gsis_id" in result.columns:
        result = result.join(
            stat_history,
            left_on="gsis_id",
            right_on="player_id",
            how="left",
        ).with_columns(pl.col("def_stat_games_observed").fill_null(0))
    else:
        result = result.with_columns(pl.lit(0, dtype=pl.Int64).alias("def_stat_games_observed"))
    return derive_defender_mechanism_signals(result)


def capability_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.len().alias("roster_rows"),
        (pl.col("defense_games_observed") > 0).sum().alias("players_with_pfr_defender_history"),
        (pl.col("def_stat_games_observed") > 0).sum().alias("players_with_def_stat_history"),
        pl.col("observed_pass_rush_signal").is_not_null().sum().alias("pass_rush_signal_players"),
        pl.col("observed_coverage_signal").is_not_null().sum().alias("coverage_signal_players"),
        pl.col("observed_run_defense_signal").is_not_null().sum().alias("run_defense_signal_players"),
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
