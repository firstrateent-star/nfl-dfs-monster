from __future__ import annotations

import polars as pl

from monster.feature_compile.depth import depth_role_multiplier

_STATUS_ACTIVE_PROB = {
    "ACT": 0.985,
    "DEV": 0.04,
    "RES": 0.01,
    "EXE": 0.0,
    "CUT": 0.0,
    "RET": 0.0,
}

_OFFENSE_DEFAULT = {
    "QB": 0.32,
    "RB": 0.24,
    "WR": 0.36,
    "TE": 0.30,
    "OL": 0.58,
}
_DEFENSE_DEFAULT = {
    "DL": 0.30,
    "LB": 0.34,
    "DB": 0.38,
}
_SPECIAL_DEFAULT = {
    "SPEC": 0.72,
    "DB": 0.16,
    "LB": 0.20,
    "WR": 0.08,
    "RB": 0.10,
    "TE": 0.10,
    "DL": 0.06,
    "OL": 0.05,
}


def _map_float(column: str, mapping: dict[str, float], default: float = 0.0) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).replace_strict(mapping, default=default).cast(pl.Float64)


def _conditional_share(
    prior: str,
    group: str,
    defaults: dict[str, float],
    depth_rank: pl.Expr,
) -> pl.Expr:
    fallback = (_map_float(group, defaults) * depth_role_multiplier(depth_rank)).clip(0.0, 0.95)
    return (
        pl.when(pl.col(prior).is_not_null())
        .then(pl.col(prior).cast(pl.Float64).clip(0.0, 1.0))
        .otherwise(fallback)
    )


def _conserve_unit(frame: pl.DataFrame, raw_column: str, output_column: str) -> pl.DataFrame:
    if "team_id" not in frame.columns:
        return frame.with_columns(pl.col(raw_column).alias(output_column))
    total = pl.col(raw_column).sum().over("team_id").clip(lower_bound=0.01)
    return frame.with_columns(
        (pl.col(raw_column) * (11.0 / total)).clip(0.0, 0.995).alias(output_column)
    )


def infer_game_day_participation(snapshot: pl.DataFrame, season: int) -> pl.DataFrame:
    """Infer roster availability and role while preserving health as a separate state."""
    required = {
        "status",
        "position_group",
        "snap_games_observed",
        "offense_snap_share",
        "defense_snap_share",
        "special_teams_snap_share",
        "changed_team_since_snap_history",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"Participation inference missing columns: {sorted(missing)}")

    if "depth_rank" not in snapshot.columns:
        snapshot = snapshot.with_columns(pl.lit(None, dtype=pl.Int64).alias("depth_rank"))
    if "health_availability_probability" not in snapshot.columns:
        snapshot = snapshot.with_columns(
            pl.lit(1.0).alias("health_availability_probability"),
            pl.lit(1.0).alias("health_effectiveness_if_active"),
            pl.lit(0.08).alias("health_uncertainty"),
        )
    depth_rank = pl.col("depth_rank").cast(pl.Int64, strict=False)

    frame = snapshot.with_columns(
        _map_float("status", _STATUS_ACTIVE_PROB).alias("roster_active_probability"),
        _conditional_share(
            "offense_snap_share", "position_group", _OFFENSE_DEFAULT, depth_rank
        ).alias("conditional_offense_snap_share"),
        _conditional_share(
            "defense_snap_share", "position_group", _DEFENSE_DEFAULT, depth_rank
        ).alias("conditional_defense_snap_share"),
        _conditional_share(
            "special_teams_snap_share", "position_group", _SPECIAL_DEFAULT, depth_rank
        ).alias("conditional_special_teams_snap_share"),
    ).with_columns(
        (
            pl.col("roster_active_probability")
            * pl.col("health_availability_probability").fill_null(1.0).clip(0.0, 1.0)
        ).clip(0.0, 1.0).alias("game_day_active_probability")
    )

    rookie_expr = (
        pl.when(pl.col("rookie_year").is_not_null())
        .then(pl.col("rookie_year").cast(pl.Int64) >= season)
        .otherwise(False)
    )
    no_history = pl.col("snap_games_observed") == 0
    changed_team = pl.col("changed_team_since_snap_history").fill_null(False)
    practice_squad = pl.col("status") == "DEV"
    has_depth = pl.col("depth_rank").is_not_null()

    frame = frame.with_columns(
        (
            0.08
            + pl.when(no_history).then(0.20).otherwise(0.0)
            + pl.when(changed_team).then(0.15).otherwise(0.0)
            + pl.when(rookie_expr).then(0.15).otherwise(0.0)
            + pl.when(practice_squad).then(0.15).otherwise(0.0)
            + pl.when(no_history & has_depth & (pl.col("depth_rank") == 1))
            .then(-0.07)
            .otherwise(0.0)
            + pl.when(no_history & has_depth & (pl.col("depth_rank") >= 3))
            .then(0.06)
            .otherwise(0.0)
            + pl.col("snap_share_uncertainty").fill_null(0.0).clip(0.0, 0.30)
        )
        .clip(0.05, 0.60)
        .alias("participation_uncertainty")
    )

    frame = frame.with_columns(
        (pl.col("game_day_active_probability") * pl.col("conditional_offense_snap_share")).alias(
            "raw_projected_offense_snap_share"
        ),
        (pl.col("game_day_active_probability") * pl.col("conditional_defense_snap_share")).alias(
            "raw_projected_defense_snap_share"
        ),
        (
            pl.col("game_day_active_probability")
            * pl.col("conditional_special_teams_snap_share")
        ).alias("raw_projected_special_teams_snap_share"),
    )
    frame = _conserve_unit(frame, "raw_projected_offense_snap_share", "projected_offense_snap_share")
    frame = _conserve_unit(frame, "raw_projected_defense_snap_share", "projected_defense_snap_share")
    frame = _conserve_unit(
        frame, "raw_projected_special_teams_snap_share", "projected_special_teams_snap_share"
    )
    frame = frame.with_columns(
        pl.max_horizontal(
            "projected_offense_snap_share",
            "projected_defense_snap_share",
            "projected_special_teams_snap_share",
        ).alias("projected_primary_snap_share")
    )

    return frame.with_columns(
        pl.when(pl.col("projected_primary_snap_share") >= 0.60)
        .then(pl.lit("core"))
        .when(pl.col("projected_primary_snap_share") >= 0.25)
        .then(pl.lit("rotation"))
        .when(pl.col("projected_primary_snap_share") >= 0.05)
        .then(pl.lit("fringe"))
        .otherwise(pl.lit("background"))
        .alias("participation_tier"),
        pl.when(pl.col("snap_games_observed") > 0)
        .then(pl.lit("recent_snaps"))
        .when(pl.col("depth_rank").is_not_null())
        .then(pl.lit("depth_prior"))
        .when(rookie_expr)
        .then(pl.lit("rookie_prior"))
        .otherwise(pl.lit("roster_prior"))
        .alias("participation_evidence"),
    )


def participation_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        (pl.col("status") == "ACT").sum().alias("active_roster_rows"),
        (pl.col("participation_tier") == "core").sum().alias("core_players"),
        (pl.col("participation_tier") == "rotation").sum().alias("rotation_players"),
        (pl.col("participation_tier") == "fringe").sum().alias("fringe_players"),
        (pl.col("participation_tier") == "background").sum().alias("background_players"),
        pl.col("projected_offense_snap_share").sum().alias("offense_snap_equivalents"),
        pl.col("projected_defense_snap_share").sum().alias("defense_snap_equivalents"),
        pl.col("projected_special_teams_snap_share").sum().alias("special_teams_snap_equivalents"),
        pl.col("participation_uncertainty").mean().alias("mean_participation_uncertainty"),
        pl.col("health_uncertainty").mean().alias("mean_health_uncertainty"),
    ]
    if "depth_rank" in snapshot.columns:
        expressions.extend(
            [
                pl.col("depth_rank").is_not_null().sum().alias("players_with_depth_role"),
                ((pl.col("snap_games_observed") == 0) & (pl.col("depth_rank") == 1))
                .sum()
                .alias("no_history_depth_starters"),
            ]
        )
    return snapshot.group_by("team_id").agg(*expressions).sort("team_id")
