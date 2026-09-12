from __future__ import annotations

import polars as pl

from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY


def _float_series(frame: pl.DataFrame, column: str) -> pl.Series:
    if column not in frame.columns:
        return pl.Series(column, [], dtype=pl.Float64)
    return frame.get_column(column).cast(pl.Float64, strict=False).drop_nulls()


def _mean(frame: pl.DataFrame, column: str, default: float) -> float:
    series = _float_series(frame, column)
    if not len(series):
        return default
    value = series.mean()
    return default if value is None else float(value)


def _sd(frame: pl.DataFrame, column: str, default: float) -> float:
    series = _float_series(frame, column)
    if len(series) < 2:
        return default
    value = series.std(ddof=1)
    return default if value is None else float(value)


def _rate(frame: pl.DataFrame, expression: pl.Expr, default: float) -> float:
    if not frame.height:
        return default
    try:
        return float(frame.select(expression.cast(pl.Float64).mean()).item())
    except (pl.exceptions.ColumnNotFoundError, TypeError, ValueError):
        return default


def _return_stats(
    frame: pl.DataFrame,
    *,
    mean_default: float,
    sd_default: float,
    zero_default: float,
    forty_default: float,
) -> tuple[float, float, float, float, int]:
    if not frame.height or "return_yards" not in frame.columns:
        return mean_default, sd_default, zero_default, forty_default, 0
    yards = frame.filter(pl.col("return_yards").is_not_null())
    if not yards.height:
        return mean_default, sd_default, zero_default, forty_default, 0
    return (
        _mean(yards, "return_yards", mean_default),
        _sd(yards, "return_yards", sd_default),
        _rate(yards, pl.col("return_yards") <= 0.0, zero_default),
        _rate(yards, pl.col("return_yards") >= 40.0, forty_default),
        yards.height,
    )


def compile_chaos_ecology(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile league-level rare-event and return priors from historical NFL play-by-play.

    The compiler is deliberately tolerant of provider-column drift. Missing optional fields
    leave the corresponding runtime default intact rather than manufacturing a zero. That
    keeps the model usable while preserving a clean seam for richer return/recovery evidence.
    """
    d = DEFAULT_CHAOS_ECOLOGY
    interception = (
        pbp.filter(pl.col("interception").fill_null(0).cast(pl.Int64) == 1)
        if "interception" in pbp.columns
        else pl.DataFrame()
    )
    fumble = (
        pbp.filter(pl.col("fumble_lost").fill_null(0).cast(pl.Int64) == 1)
        if "fumble_lost" in pbp.columns
        else pl.DataFrame()
    )
    punt = (
        pbp.filter(pl.col("play_type") == "punt")
        if "play_type" in pbp.columns
        else pl.DataFrame()
    )
    kickoff = (
        pbp.filter(pl.col("play_type") == "kickoff")
        if "play_type" in pbp.columns
        else pl.DataFrame()
    )

    int_mean, int_sd, int_zero, int_40, int_rows = _return_stats(
        interception,
        mean_default=d.interception_return_mean,
        sd_default=d.interception_return_sd,
        zero_default=d.interception_zero_return_rate,
        forty_default=d.interception_40_plus_rate,
    )
    fum_mean, fum_sd, fum_zero, fum_40, fum_rows = _return_stats(
        fumble,
        mean_default=d.fumble_return_mean,
        sd_default=d.fumble_return_sd,
        zero_default=d.fumble_zero_return_rate,
        forty_default=d.fumble_40_plus_rate,
    )
    punt_mean, punt_sd, punt_zero, punt_40, punt_rows = _return_stats(
        punt,
        mean_default=d.punt_return_mean,
        sd_default=d.punt_return_sd,
        zero_default=d.punt_zero_return_rate,
        forty_default=d.punt_40_plus_rate,
    )
    kick_mean, kick_sd, _, kick_40, kick_rows = _return_stats(
        kickoff,
        mean_default=d.kickoff_return_mean,
        sd_default=d.kickoff_return_sd,
        zero_default=0.0,
        forty_default=d.kickoff_40_plus_rate,
    )

    punt_blocked = d.blocked_punt_rate
    if punt.height and "punt_blocked" in punt.columns:
        punt_blocked = _rate(punt, pl.col("punt_blocked").fill_null(0) == 1, punt_blocked)

    fg = (
        pbp.filter(pl.col("field_goal_attempt").fill_null(0).cast(pl.Int64) == 1)
        if "field_goal_attempt" in pbp.columns
        else pl.DataFrame()
    )
    fg_blocked = d.blocked_field_goal_rate
    if fg.height and "field_goal_result" in fg.columns:
        fg_blocked = _rate(
            fg,
            pl.col("field_goal_result").cast(pl.Utf8).str.to_lowercase() == "blocked",
            fg_blocked,
        )

    # A return_yards null/zero kickoff is only a loose proxy for a touchback because provider
    # conventions differ; use it only when explicit kickoff rows provide enough evidence.
    touchback = d.kickoff_touchback_rate
    if kickoff.height and "return_yards" in kickoff.columns:
        returned = kickoff.get_column("return_yards").is_not_null()
        touchback = float((~returned).mean())

    return pl.DataFrame(
        [
            {
                "interception_zero_return_rate": int_zero,
                "interception_return_mean": int_mean,
                "interception_return_sd": int_sd,
                "interception_40_plus_rate": int_40,
                "interception_return_rows": int_rows,
                "fumble_zero_return_rate": fum_zero,
                "fumble_return_mean": fum_mean,
                "fumble_return_sd": fum_sd,
                "fumble_40_plus_rate": fum_40,
                "fumble_return_rows": fum_rows,
                "punt_zero_return_rate": punt_zero,
                "punt_return_mean": punt_mean,
                "punt_return_sd": punt_sd,
                "punt_40_plus_rate": punt_40,
                "punt_return_rows": punt_rows,
                "kickoff_return_mean": kick_mean,
                "kickoff_return_sd": kick_sd,
                "kickoff_40_plus_rate": kick_40,
                "kickoff_return_rows": kick_rows,
                "punt_muff_rate": d.punt_muff_rate,
                "punt_muff_kicking_recovery_rate": d.punt_muff_kicking_recovery_rate,
                "kickoff_muff_rate": d.kickoff_muff_rate,
                "kickoff_muff_kicking_recovery_rate": d.kickoff_muff_kicking_recovery_rate,
                "blocked_punt_rate": punt_blocked,
                "blocked_field_goal_rate": fg_blocked,
                "kickoff_touchback_rate": touchback,
                "kickoff_touchback_yardline": 35.0,
            }
        ]
    )
