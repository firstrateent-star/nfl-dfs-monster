from __future__ import annotations

import polars as pl


SNAP_COLUMNS = (
    "offense_pct",
    "defense_pct",
    "st_pct",
)


def _normalize_pct(expr: pl.Expr) -> pl.Expr:
    """Normalize snap percentage fields whether source stores 0-1 or 0-100."""
    return (
        pl.when(expr > 1.0)
        .then(expr / 100.0)
        .otherwise(expr)
        .clip(lower_bound=0.0, upper_bound=1.0)
    )


def compile_snap_priors(snap_counts: pl.DataFrame, recent_games: int = 6) -> pl.DataFrame:
    """Compile player field-time priors from recent game-level snap counts.

    Snap share is a *weight on every other player metric*. It is not itself an
    efficiency bonus. For a Week 1 slate the prior naturally falls back to the
    player's most recent games from the previous season; current depth chart,
    injury and role evidence can update the prior later in the snapshot compiler.
    """
    required = {
        "season",
        "week",
        "team",
        "position",
        "pfr_player_id",
        "offense_pct",
        "defense_pct",
        "st_pct",
    }
    missing = required.difference(snap_counts.columns)
    if missing:
        raise ValueError(f"Snap counts missing required columns: {sorted(missing)}")

    ordered = snap_counts.sort(["season", "week"]).with_columns(
        _normalize_pct(pl.col("offense_pct").cast(pl.Float64)).alias("offense_pct_norm"),
        _normalize_pct(pl.col("defense_pct").cast(pl.Float64)).alias("defense_pct_norm"),
        _normalize_pct(pl.col("st_pct").cast(pl.Float64)).alias("st_pct_norm"),
    )

    return (
        ordered.group_by(["team", "pfr_player_id"], maintain_order=True)
        .agg(
            pl.col("position").last().alias("position"),
            pl.col("offense_pct_norm").tail(recent_games).mean().alias("offense_snap_share"),
            pl.col("defense_pct_norm").tail(recent_games).mean().alias("defense_snap_share"),
            pl.col("st_pct_norm").tail(recent_games).mean().alias("special_teams_snap_share"),
            pl.col("offense_pct_norm").tail(recent_games).std().fill_null(0.0).alias("offense_snap_uncertainty"),
            pl.col("defense_pct_norm").tail(recent_games).std().fill_null(0.0).alias("defense_snap_uncertainty"),
            pl.col("st_pct_norm").tail(recent_games).std().fill_null(0.0).alias("special_teams_snap_uncertainty"),
            pl.len().clip(upper_bound=recent_games).alias("snap_games_observed"),
        )
        .with_columns(
            pl.max_horizontal(
                "offense_snap_share",
                "defense_snap_share",
                "special_teams_snap_share",
            ).alias("primary_snap_share"),
            pl.max_horizontal(
                "offense_snap_uncertainty",
                "defense_snap_uncertainty",
                "special_teams_snap_uncertainty",
            ).alias("snap_share_uncertainty"),
        )
    )
