from __future__ import annotations

import polars as pl

_OL_GROUP = "OL"


def _require(frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _z_signal(frame: pl.DataFrame, column: str, *, reverse: bool = False) -> pl.Expr:
    mean = float(frame.get_column(column).mean() or 0.0)
    std = float(frame.get_column(column).std() or 0.0)
    if std < 1e-9:
        return pl.lit(0.0)
    expr = ((pl.col(column) - mean) / std).clip(-2.5, 2.5) / 2.5
    return -expr if reverse else expr


def compile_historical_ol_outcomes(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile team-level OL outcome priors from market-blind play-by-play.

    These are *unit outcome* priors, not individual grades. Pass protection uses sack
    and QB-hit frequency. Run blocking uses rushing EPA/success/stuff/explosive rates.
    Current personnel continuity later determines how much of this historical unit
    evidence is allowed to survive into the new season.
    """
    required = {
        "posteam",
        "play_type",
        "qb_dropback",
        "sack",
        "qb_hit",
        "rush_attempt",
        "epa",
        "success",
        "yards_gained",
    }
    _require(pbp, required, "OL outcome PBP")

    pass_plays = pbp.filter(
        pl.col("posteam").is_not_null() & (pl.col("qb_dropback") == 1)
    )
    pass_team = (
        pass_plays.group_by("posteam")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("sack").sum().alias("sacks_allowed"),
            pl.col("qb_hit").sum().alias("qb_hits_allowed"),
        )
        .with_columns(
            (pl.col("sacks_allowed") / pl.col("dropbacks").clip(lower_bound=1)).alias(
                "sack_rate_allowed"
            ),
            (pl.col("qb_hits_allowed") / pl.col("dropbacks").clip(lower_bound=1)).alias(
                "qb_hit_rate_allowed"
            ),
        )
        .rename({"posteam": "team_id"})
    )

    rush_plays = pbp.filter(
        pl.col("posteam").is_not_null()
        & (pl.col("rush_attempt") == 1)
        & (pl.col("play_type") == "run")
    )
    rush_team = (
        rush_plays.with_columns(
            (pl.col("yards_gained") <= 0).cast(pl.Float64).alias("stuff"),
            (pl.col("yards_gained") >= 10).cast(pl.Float64).alias("explosive_rush"),
        )
        .group_by("posteam")
        .agg(
            pl.len().alias("rushes"),
            pl.col("epa").mean().alias("rush_epa_per_play"),
            pl.col("success").mean().alias("rush_success_rate"),
            pl.col("stuff").mean().alias("stuff_rate"),
            pl.col("explosive_rush").mean().alias("explosive_rush_rate"),
        )
        .rename({"posteam": "team_id"})
    )

    out = pass_team.join(rush_team, on="team_id", how="full", coalesce=True).sort("team_id")
    if not out.height:
        return out

    out = out.with_columns(
        (
            0.62 * _z_signal(out, "sack_rate_allowed", reverse=True)
            + 0.38 * _z_signal(out, "qb_hit_rate_allowed", reverse=True)
        )
        .clip(-1.0, 1.0)
        .alias("historical_pass_protection_signal"),
        (
            0.42 * _z_signal(out, "rush_epa_per_play")
            + 0.28 * _z_signal(out, "rush_success_rate")
            + 0.18 * _z_signal(out, "stuff_rate", reverse=True)
            + 0.12 * _z_signal(out, "explosive_rush_rate")
        )
        .clip(-1.0, 1.0)
        .alias("historical_run_block_signal"),
    )
    return out


def compile_current_ol_context(
    personnel: pl.DataFrame,
    historical: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Measure how much historical OL evidence still belongs to the current line.

    Historical unit outcomes are shrunk toward neutral when current linemen lack snap
    linkage or changed teams. Missing evidence increases uncertainty; it is never treated
    as poor performance.
    """
    required = {
        "team_id",
        "position_group",
        "projected_offense_snap_share",
        "participation_uncertainty",
        "snap_games_observed",
        "prior_team_id",
    }
    _require(personnel, required, "Current OL personnel")

    ol = personnel.filter(pl.col("position_group") == _OL_GROUP).with_columns(
        pl.col("projected_offense_snap_share").cast(pl.Float64).clip(0.0, 1.0).alias("_w"),
        (
            (pl.col("snap_games_observed") > 0)
            & pl.col("prior_team_id").is_not_null()
            & (pl.col("prior_team_id") == pl.col("team_id"))
        )
        .cast(pl.Float64)
        .alias("_returning"),
        (pl.col("snap_games_observed") > 0).cast(pl.Float64).alias("_linked"),
    )

    context = (
        ol.group_by("team_id")
        .agg(
            pl.col("_w").sum().alias("projected_ol_snap_equivalents"),
            (pl.col("_w") * pl.col("_linked")).sum().alias("linked_ol_snap_weight"),
            (pl.col("_w") * pl.col("_returning")).sum().alias("returning_ol_snap_weight"),
            (
                (pl.col("_w") * pl.col("participation_uncertainty")).sum()
                / pl.col("_w").sum().clip(lower_bound=0.01)
            ).alias("weighted_ol_participation_uncertainty"),
            pl.len().alias("ol_roster_rows"),
        )
        .with_columns(
            (
                pl.col("linked_ol_snap_weight")
                / pl.col("projected_ol_snap_equivalents").clip(lower_bound=0.01)
            )
            .clip(0.0, 1.0)
            .alias("ol_history_coverage"),
            (
                pl.col("returning_ol_snap_weight")
                / pl.col("projected_ol_snap_equivalents").clip(lower_bound=0.01)
            )
            .clip(0.0, 1.0)
            .alias("ol_continuity"),
        )
        .with_columns(
            (
                pl.col("ol_history_coverage")
                * (0.35 + 0.65 * pl.col("ol_continuity"))
            )
            .clip(0.0, 1.0)
            .alias("historical_ol_authority"),
            (
                pl.col("weighted_ol_participation_uncertainty")
                + 0.18 * (1.0 - pl.col("ol_history_coverage"))
                + 0.12 * (1.0 - pl.col("ol_continuity"))
            )
            .clip(0.05, 0.65)
            .alias("ol_context_uncertainty"),
        )
        .sort("team_id")
    )

    if historical is None or not historical.height:
        return context.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("historical_pass_protection_signal"),
            pl.lit(None, dtype=pl.Float64).alias("historical_run_block_signal"),
            pl.lit(None, dtype=pl.Float64).alias("observed_pass_block_signal"),
            pl.lit(None, dtype=pl.Float64).alias("observed_run_block_signal"),
        )

    joined = context.join(
        historical.select(
            [
                "team_id",
                "historical_pass_protection_signal",
                "historical_run_block_signal",
            ]
        ),
        on="team_id",
        how="left",
    )
    return joined.with_columns(
        (
            pl.col("historical_pass_protection_signal") * pl.col("historical_ol_authority")
        )
        .clip(-1.0, 1.0)
        .alias("observed_pass_block_signal"),
        (pl.col("historical_run_block_signal") * pl.col("historical_ol_authority"))
        .clip(-1.0, 1.0)
        .alias("observed_run_block_signal"),
    )


def attach_current_ol_context(personnel: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Attach current-line observed signals only to players with OL jurisdiction."""
    columns = [
        "team_id",
        "observed_pass_block_signal",
        "observed_run_block_signal",
        "ol_continuity",
        "ol_context_uncertainty",
        "historical_ol_authority",
    ]
    available = [c for c in columns if c in context.columns]
    out = personnel.join(context.select(available), on="team_id", how="left")
    is_ol = pl.col("position_group") == _OL_GROUP
    return out.with_columns(
        pl.when(is_ol)
        .then(pl.col("observed_pass_block_signal"))
        .otherwise(None)
        .alias("observed_pass_block_signal"),
        pl.when(is_ol)
        .then(pl.col("observed_run_block_signal"))
        .otherwise(None)
        .alias("observed_run_block_signal"),
    )
