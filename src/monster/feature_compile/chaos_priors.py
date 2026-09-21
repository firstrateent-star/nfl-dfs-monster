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
    yard_column: str = "return_yards",
    mean_default: float,
    sd_default: float,
    zero_default: float,
    twenty_default: float,
    forty_default: float,
    sixty_default: float,
    eighty_default: float,
) -> tuple[float, float, float, float, float, float, float, int]:
    defaults = (
        mean_default,
        sd_default,
        zero_default,
        twenty_default,
        forty_default,
        sixty_default,
        eighty_default,
        0,
    )
    if not frame.height or yard_column not in frame.columns:
        return defaults
    yards = frame.filter(pl.col(yard_column).is_not_null())
    if not yards.height:
        return defaults
    return (
        _mean(yards, yard_column, mean_default),
        _sd(yards, yard_column, sd_default),
        _rate(yards, pl.col(yard_column) <= 0.0, zero_default),
        _rate(yards, pl.col(yard_column) >= 20.0, twenty_default),
        _rate(yards, pl.col(yard_column) >= 40.0, forty_default),
        _rate(yards, pl.col(yard_column) >= 60.0, sixty_default),
        _rate(yards, pl.col(yard_column) >= 80.0, eighty_default),
        yards.height,
    )


def _flag(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64, strict=False) == 1


def _present(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).is_not_null()


def _fraction(numerator: int, denominator: int, default: float) -> float:
    return default if denominator <= 0 else float(numerator / denominator)



def _field_goal_bucket(
    frame: pl.DataFrame,
    *,
    low: float,
    high: float | None,
    default_make: float,
) -> tuple[int, int, float]:
    """Compile conditional make probability after excluding blocked attempts.

    Block probability remains a separate live-ball mechanism. This prevents a blocked kick
    from being counted once in the blocked branch and again inside ordinary make probability.
    """
    if (
        not frame.height
        or "kick_distance" not in frame.columns
        or "field_goal_result" not in frame.columns
    ):
        return 0, 0, default_make
    distance = pl.col("kick_distance").cast(pl.Float64, strict=False)
    condition = distance >= low
    if high is not None:
        condition = condition & (distance <= high)
    bucket = frame.filter(condition & distance.is_not_null())
    if not bucket.height:
        return 0, 0, default_make
    result = pl.col("field_goal_result").cast(pl.Utf8).str.to_lowercase()
    unblocked = bucket.filter(result != "blocked")
    if not unblocked.height:
        return bucket.height, 0, default_make
    make_rate = float(
        unblocked.select((result == "made").cast(pl.Float64).mean()).item()
    )
    # Sparse current-week evidence cannot seize authority from the full prior-season sample.
    weight = float(unblocked.height / (unblocked.height + 40.0))
    rate = default_make + weight * (make_rate - default_make)
    return bucket.height, unblocked.height, float(min(max(rate, 0.05), 0.995))

def compile_chaos_ecology(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile rare-event and return priors without mixing return selection and distance.

    Stable legacy columns remain unchanged for v6.2/v6.3 controls. New ``*_live_*`` and
    ``fumble_recovery_*`` columns describe the causal subchannels required by v6.3.2:
    whether a live return occurs and, conditional on that state, how far it travels. No
    touchdown-rate field is compiled; the live field still decides whether sampled distance scores.
    """
    d = DEFAULT_CHAOS_ECOLOGY
    interception = (
        pbp.filter(_flag(pbp, "interception")) if "interception" in pbp.columns else pl.DataFrame()
    )
    fumble = pbp.filter(_flag(pbp, "fumble_lost")) if "fumble_lost" in pbp.columns else pl.DataFrame()
    punt = pbp.filter(pl.col("play_type") == "punt") if "play_type" in pbp.columns else pl.DataFrame()
    kickoff = (
        pbp.filter(pl.col("play_type") == "kickoff") if "play_type" in pbp.columns else pl.DataFrame()
    )

    int_mean, int_sd, int_zero, int_20, int_40, int_60, int_80, int_rows = _return_stats(
        interception,
        mean_default=d.interception_return_mean,
        sd_default=d.interception_return_sd,
        zero_default=d.interception_zero_return_rate,
        twenty_default=d.interception_20_plus_rate,
        forty_default=d.interception_40_plus_rate,
        sixty_default=d.interception_60_plus_rate,
        eighty_default=d.interception_80_plus_rate,
    )
    fum_mean, fum_sd, fum_zero, fum_20, fum_40, fum_60, fum_80, fum_rows = _return_stats(
        fumble,
        mean_default=d.fumble_return_mean,
        sd_default=d.fumble_return_sd,
        zero_default=d.fumble_zero_return_rate,
        twenty_default=d.fumble_20_plus_rate,
        forty_default=d.fumble_40_plus_rate,
        sixty_default=d.fumble_60_plus_rate,
        eighty_default=d.fumble_80_plus_rate,
    )
    punt_mean, punt_sd, punt_zero, punt_20, punt_40, punt_60, punt_80, punt_rows = _return_stats(
        punt,
        mean_default=d.punt_return_mean,
        sd_default=d.punt_return_sd,
        zero_default=d.punt_zero_return_rate,
        twenty_default=d.punt_20_plus_rate,
        forty_default=d.punt_40_plus_rate,
        sixty_default=d.punt_60_plus_rate,
        eighty_default=d.punt_80_plus_rate,
    )
    kick_mean, kick_sd, _, kick_20, kick_40, kick_60, kick_80, kick_rows = _return_stats(
        kickoff,
        mean_default=d.kickoff_return_mean,
        sd_default=d.kickoff_return_sd,
        zero_default=0.0,
        twenty_default=d.kickoff_20_plus_rate,
        forty_default=d.kickoff_40_plus_rate,
        sixty_default=d.kickoff_60_plus_rate,
        eighty_default=d.kickoff_80_plus_rate,
    )

    # Fumble returns live in recovery-specific nflverse fields. Generic return_yards suppresses
    # these events and was the main reason the original fumble-return ecology could not create
    # realistic defensive scores.
    (
        fum_rec_mean,
        fum_rec_sd,
        fum_rec_zero,
        fum_rec_20,
        fum_rec_40,
        fum_rec_60,
        fum_rec_80,
        fum_rec_rows,
    ) = _return_stats(
        fumble,
        yard_column="fumble_recovery_1_yards",
        mean_default=3.5,
        sd_default=12.0,
        zero_default=0.76,
        twenty_default=0.065,
        forty_default=0.016,
        sixty_default=0.012,
        eighty_default=0.004,
    )

    punt_blocked = d.blocked_punt_rate
    if punt.height and "punt_blocked" in punt.columns:
        punt_blocked = _rate(punt, _flag(punt, "punt_blocked"), punt_blocked)
    punt_touchback_explicit = _rate(punt, _flag(punt, "touchback"), 0.08)
    punt_eligible = punt.filter(~_flag(punt, "punt_blocked") & ~_flag(punt, "touchback"))
    punt_live = punt_eligible.filter(
        _present(punt_eligible, "punt_returner_player_id") & ~_flag(punt_eligible, "punt_fair_catch")
    )
    punt_live_rate = _fraction(punt_live.height, punt_eligible.height, 0.47)
    (
        punt_live_mean,
        punt_live_sd,
        punt_live_zero,
        punt_live_20,
        punt_live_40,
        punt_live_60,
        punt_live_80,
        punt_live_rows,
    ) = _return_stats(
        punt_live,
        mean_default=10.2,
        sd_default=13.0,
        zero_default=0.19,
        twenty_default=0.094,
        forty_default=0.036,
        sixty_default=0.019,
        eighty_default=0.007,
    )

    kickoff_touchback_explicit = _rate(kickoff, _flag(kickoff, "touchback"), 0.21)
    kickoff_eligible = kickoff.filter(~_flag(kickoff, "touchback"))
    kickoff_live = kickoff_eligible.filter(_present(kickoff_eligible, "kickoff_returner_player_id"))
    kickoff_live_rate = _fraction(kickoff_live.height, kickoff_eligible.height, 0.94)
    (
        kick_live_mean,
        kick_live_sd,
        kick_live_zero,
        kick_live_20,
        kick_live_40,
        kick_live_60,
        kick_live_80,
        kick_live_rows,
    ) = _return_stats(
        kickoff_live,
        mean_default=25.9,
        sd_default=10.5,
        zero_default=0.006,
        twenty_default=0.862,
        forty_default=0.042,
        sixty_default=0.009,
        eighty_default=0.005,
    )

    fg = (
        pbp.filter(_flag(pbp, "field_goal_attempt"))
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

    fg_buckets = {
        "00_29": _field_goal_bucket(fg, low=0.0, high=29.0, default_make=0.985),
        "30_39": _field_goal_bucket(fg, low=30.0, high=39.0, default_make=0.955),
        "40_49": _field_goal_bucket(fg, low=40.0, high=49.0, default_make=0.885),
        "50_59": _field_goal_bucket(fg, low=50.0, high=59.0, default_make=0.735),
        "60_plus": _field_goal_bucket(fg, low=60.0, high=None, default_make=0.480),
    }

    # Legacy field retained so prior runners remain bit-for-bit compatible with their previous
    # policy semantics. v6.3.2 consumes kickoff_touchback_rate_explicit instead.
    legacy_touchback = d.kickoff_touchback_rate
    if kickoff.height and "return_yards" in kickoff.columns:
        returned = kickoff.get_column("return_yards").is_not_null()
        legacy_touchback = float((~returned).mean())

    return pl.DataFrame(
        [
            {
                "interception_zero_return_rate": int_zero,
                "interception_return_mean": int_mean,
                "interception_return_sd": int_sd,
                "interception_20_plus_rate": int_20,
                "interception_40_plus_rate": int_40,
                "interception_60_plus_rate": int_60,
                "interception_80_plus_rate": int_80,
                "interception_return_rows": int_rows,
                "fumble_zero_return_rate": fum_zero,
                "fumble_return_mean": fum_mean,
                "fumble_return_sd": fum_sd,
                "fumble_20_plus_rate": fum_20,
                "fumble_40_plus_rate": fum_40,
                "fumble_60_plus_rate": fum_60,
                "fumble_80_plus_rate": fum_80,
                "fumble_return_rows": fum_rows,
                "fumble_recovery_zero_return_rate": fum_rec_zero,
                "fumble_recovery_return_mean": fum_rec_mean,
                "fumble_recovery_return_sd": fum_rec_sd,
                "fumble_recovery_20_plus_rate": fum_rec_20,
                "fumble_recovery_40_plus_rate": fum_rec_40,
                "fumble_recovery_60_plus_rate": fum_rec_60,
                "fumble_recovery_80_plus_rate": fum_rec_80,
                "fumble_recovery_return_rows": fum_rec_rows,
                "punt_zero_return_rate": punt_zero,
                "punt_return_mean": punt_mean,
                "punt_return_sd": punt_sd,
                "punt_20_plus_rate": punt_20,
                "punt_40_plus_rate": punt_40,
                "punt_60_plus_rate": punt_60,
                "punt_80_plus_rate": punt_80,
                "punt_return_rows": punt_rows,
                "punt_touchback_rate_explicit": punt_touchback_explicit,
                "punt_live_return_rate_after_touchback": punt_live_rate,
                "punt_live_return_zero_rate": punt_live_zero,
                "punt_live_return_mean": punt_live_mean,
                "punt_live_return_sd": punt_live_sd,
                "punt_live_20_plus_rate": punt_live_20,
                "punt_live_40_plus_rate": punt_live_40,
                "punt_live_60_plus_rate": punt_live_60,
                "punt_live_80_plus_rate": punt_live_80,
                "punt_live_return_rows": punt_live_rows,
                "kickoff_return_mean": kick_mean,
                "kickoff_return_sd": kick_sd,
                "kickoff_20_plus_rate": kick_20,
                "kickoff_40_plus_rate": kick_40,
                "kickoff_60_plus_rate": kick_60,
                "kickoff_80_plus_rate": kick_80,
                "kickoff_return_rows": kick_rows,
                "kickoff_touchback_rate_explicit": kickoff_touchback_explicit,
                "kickoff_live_return_rate_after_touchback": kickoff_live_rate,
                "kickoff_live_return_zero_rate": kick_live_zero,
                "kickoff_live_return_mean": kick_live_mean,
                "kickoff_live_return_sd": kick_live_sd,
                "kickoff_live_20_plus_rate": kick_live_20,
                "kickoff_live_40_plus_rate": kick_live_40,
                "kickoff_live_60_plus_rate": kick_live_60,
                "kickoff_live_80_plus_rate": kick_live_80,
                "kickoff_live_return_rows": kick_live_rows,
                "punt_muff_rate": d.punt_muff_rate,
                "punt_muff_kicking_recovery_rate": d.punt_muff_kicking_recovery_rate,
                "kickoff_muff_rate": d.kickoff_muff_rate,
                "kickoff_muff_kicking_recovery_rate": d.kickoff_muff_kicking_recovery_rate,
                "blocked_punt_rate": punt_blocked,
                "blocked_field_goal_rate": fg_blocked,
                "v725_fg_00_29_attempts": fg_buckets["00_29"][0],
                "v725_fg_00_29_unblocked_attempts": fg_buckets["00_29"][1],
                "v725_fg_00_29_make_rate_unblocked": fg_buckets["00_29"][2],
                "v725_fg_30_39_attempts": fg_buckets["30_39"][0],
                "v725_fg_30_39_unblocked_attempts": fg_buckets["30_39"][1],
                "v725_fg_30_39_make_rate_unblocked": fg_buckets["30_39"][2],
                "v725_fg_40_49_attempts": fg_buckets["40_49"][0],
                "v725_fg_40_49_unblocked_attempts": fg_buckets["40_49"][1],
                "v725_fg_40_49_make_rate_unblocked": fg_buckets["40_49"][2],
                "v725_fg_50_59_attempts": fg_buckets["50_59"][0],
                "v725_fg_50_59_unblocked_attempts": fg_buckets["50_59"][1],
                "v725_fg_50_59_make_rate_unblocked": fg_buckets["50_59"][2],
                "v725_fg_60_plus_attempts": fg_buckets["60_plus"][0],
                "v725_fg_60_plus_unblocked_attempts": fg_buckets["60_plus"][1],
                "v725_fg_60_plus_make_rate_unblocked": fg_buckets["60_plus"][2],
                "kickoff_touchback_rate": legacy_touchback,
                "kickoff_touchback_yardline": 35.0,
            }
        ]
    )