from __future__ import annotations

import math

import polars as pl

_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("00_20", 0.0, 20.0),
    ("20_40", 20.0, 40.0),
    ("40_60", 40.0, 60.0),
    ("60_80", 60.0, 80.0),
    ("80_100", 80.0, 100.0001),
)


def _flag(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64, strict=False) == 1


def _present(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).is_not_null()


def _float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _series_stats(frame: pl.DataFrame, column: str) -> dict[str, float | int]:
    if not frame.height or column not in frame.columns:
        return {
            "rows": 0,
            "zero": 1.0,
            "mean": 0.0,
            "sd": 1.0,
            "p5": 0.0,
            "p10": 0.0,
            "p15": 0.0,
            "p20": 0.0,
            "p40": 0.0,
            "p60": 0.0,
            "p80": 0.0,
            "upper80_rows": 0,
            "upper80_mean": 88.0,
            "upper80_sd": 7.0,
        }
    values = frame.get_column(column).cast(pl.Float64, strict=False).drop_nulls()
    if not len(values):
        return {
            "rows": 0,
            "zero": 1.0,
            "mean": 0.0,
            "sd": 1.0,
            "p5": 0.0,
            "p10": 0.0,
            "p15": 0.0,
            "p20": 0.0,
            "p40": 0.0,
            "p60": 0.0,
            "p80": 0.0,
            "upper80_rows": 0,
            "upper80_mean": 88.0,
            "upper80_sd": 7.0,
        }
    upper = values.filter(values >= 80.0)
    sd = values.std(ddof=1) if len(values) >= 2 else 1.0
    upper_sd = upper.std(ddof=1) if len(upper) >= 2 else 7.0
    return {
        "rows": int(len(values)),
        "zero": float((values <= 0.0).mean()),
        "mean": _float(values.mean()),
        "sd": max(_float(sd, 1.0), 0.5),
        "p5": float((values >= 5.0).mean()),
        "p10": float((values >= 10.0).mean()),
        "p15": float((values >= 15.0).mean()),
        "p20": float((values >= 20.0).mean()),
        "p40": float((values >= 40.0).mean()),
        "p60": float((values >= 60.0).mean()),
        "p80": float((values >= 80.0).mean()),
        "upper80_rows": int(len(upper)),
        "upper80_mean": _float(upper.mean(), 88.0) if len(upper) else 88.0,
        "upper80_sd": max(_float(upper_sd, 7.0), 1.0) if len(upper) else 7.0,
    }


def _flatten_profile(out: dict[str, object], prefix: str, stats: dict[str, float | int]) -> None:
    out[f"{prefix}_rows"] = stats["rows"]
    out[f"{prefix}_zero_rate"] = stats["zero"]
    out[f"{prefix}_mean"] = stats["mean"]
    out[f"{prefix}_sd"] = stats["sd"]
    for threshold in (5, 10, 15, 20, 40, 60, 80):
        out[f"{prefix}_p{threshold}"] = stats[f"p{threshold}"]
    out[f"{prefix}_upper80_rows"] = stats["upper80_rows"]
    out[f"{prefix}_upper80_mean"] = stats["upper80_mean"]
    out[f"{prefix}_upper80_sd"] = stats["upper80_sd"]


def _compile_bucketed(
    frame: pl.DataFrame,
    *,
    required_column: str,
    yards_column: str,
    prefix: str,
    out: dict[str, object],
) -> None:
    if not frame.height:
        for name, _, _ in _BUCKETS:
            _flatten_profile(out, f"v633_{prefix}_{name}", _series_stats(pl.DataFrame(), yards_column))
        return
    for name, lower, upper in _BUCKETS:
        if name == "00_20":
            bucket = frame.filter(
                (pl.col(required_column) >= lower) & (pl.col(required_column) <= upper)
            )
        else:
            bucket = frame.filter(
                (pl.col(required_column) > lower) & (pl.col(required_column) <= upper)
            )
        _flatten_profile(
            out,
            f"v633_{prefix}_{name}",
            _series_stats(bucket, yards_column),
        )


def _drive_starts(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {"game_id", "fixed_drive", "yardline_100"}
    if not required.issubset(pbp.columns):
        return pl.DataFrame()
    scrimmage = _flag(pbp, "pass_attempt") | _flag(pbp, "rush_attempt") | _flag(pbp, "sack")
    sort_columns = [column for column in ("game_id", "fixed_drive", "play_id") if column in pbp.columns]
    return (
        pbp.filter(
            pl.col("game_id").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & pl.col("yardline_100").is_not_null()
            & scrimmage
        )
        .sort(sort_columns)
        .group_by(["game_id", "fixed_drive"], maintain_order=True)
        .agg(pl.col("yardline_100").first().alias("first_scrimmage_yardline_100"))
        .with_columns(
            (100.0 - pl.col("first_scrimmage_yardline_100")).alias("drive_start_from_own_goal")
        )
    )


def compile_return_geometry_v633(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile field-conditioned return-distance evidence for the v6.3.3 shadow runner.

    Every output describes a physical state or a return-distance distribution. No touchdown rate
    is compiled or exposed to runtime. A simulated score still requires a sampled return distance
    to traverse the remaining field.
    """

    out: dict[str, object] = {}

    # Fumble recovery geometry. nflverse yardline_100 is distance to the offense's target goal,
    # while Monster's live field uses distance from the offense's own goal. Convert to the latter.
    if {
        "yardline_100",
        "yards_gained",
        "fumble_recovery_1_yards",
    }.issubset(pbp.columns):
        fumbles = (
            pbp.filter(_flag(pbp, "fumble_lost") & pl.col("fumble_recovery_1_yards").is_not_null())
            .with_columns(
                (
                    100.0
                    - pl.col("yardline_100")
                    + pl.col("yards_gained").fill_null(0.0)
                )
                .cast(pl.Float64)
                .clip(0.0, 100.0)
                .alias("required_return_distance"),
                pl.col("fumble_recovery_1_yards").cast(pl.Float64).alias("return_distance"),
            )
        )
    else:
        fumbles = pl.DataFrame()
    _compile_bucketed(
        fumbles,
        required_column="required_return_distance",
        yards_column="return_distance",
        prefix="fumble",
        out=out,
    )
    _flatten_profile(out, "v633_fumble_global", _series_stats(fumbles, "return_distance"))

    # Punt geometry is observable from scrimmage field position + gross kick distance.
    punt_required = {
        "yardline_100",
        "kick_distance",
        "return_yards",
        "punt_returner_player_id",
    }
    if punt_required.issubset(pbp.columns):
        punts = (
            pbp.filter(
                _flag(pbp, "punt_attempt")
                & ~_flag(pbp, "punt_blocked")
                & ~_flag(pbp, "touchback")
                & _present(pbp, "punt_returner_player_id")
                & ~_flag(pbp, "punt_fair_catch")
                & pl.col("return_yards").is_not_null()
            )
            .with_columns(
                (
                    100.0
                    - pl.col("yardline_100")
                    + pl.col("kick_distance").fill_null(0.0)
                )
                .cast(pl.Float64)
                .clip(0.0, 100.0)
                .alias("required_return_distance"),
                pl.col("return_yards").cast(pl.Float64).alias("return_distance"),
            )
        )
    else:
        punts = pl.DataFrame()
    _compile_bucketed(
        punts,
        required_column="required_return_distance",
        yards_column="return_distance",
        prefix="punt",
        out=out,
    )
    _flatten_profile(out, "v633_punt_global", _series_stats(punts, "return_distance"))

    # Dynamic-kickoff kick_distance does not reliably encode receiving landing position. For
    # ordinary (non-TD) returns, reconstruct landing = receiving drive start - return distance.
    drive_starts = _drive_starts(pbp)
    kickoff_required = {"game_id", "fixed_drive", "return_yards", "kickoff_returner_player_id"}
    if kickoff_required.issubset(pbp.columns) and drive_starts.height:
        kickoff_live = pbp.filter(
            _flag(pbp, "kickoff_attempt")
            & ~_flag(pbp, "touchback")
            & _present(pbp, "kickoff_returner_player_id")
            & pl.col("return_yards").is_not_null()
        )
        ordinary = (
            kickoff_live.join(drive_starts, on=["game_id", "fixed_drive"], how="inner")
            .with_columns(
                (
                    pl.col("drive_start_from_own_goal")
                    - pl.col("return_yards").cast(pl.Float64)
                )
                .clip(0.0, 20.0)
                .alias("landing_from_receiving_goal")
            )
        )
        landings = ordinary.get_column("landing_from_receiving_goal").drop_nulls()
        out["v633_kickoff_landing_rows"] = int(len(landings))
        out["v633_kickoff_landing_mean"] = _float(landings.mean(), 4.2) if len(landings) else 4.2
        landing_sd = landings.std(ddof=1) if len(landings) >= 2 else 3.5
        out["v633_kickoff_landing_sd"] = max(_float(landing_sd, 3.5), 0.5)
        out["v633_kickoff_landing_p25"] = _float(landings.quantile(0.25), 1.0) if len(landings) else 1.0
        out["v633_kickoff_landing_p50"] = _float(landings.quantile(0.50), 3.0) if len(landings) else 3.0
        out["v633_kickoff_landing_p75"] = _float(landings.quantile(0.75), 6.0) if len(landings) else 6.0
        out["v633_kickoff_returned_drive_start_mean"] = (
            _float(ordinary.get_column("drive_start_from_own_goal").mean(), 29.6)
            if ordinary.height
            else 29.6
        )
        _flatten_profile(
            out,
            "v633_kickoff_global",
            _series_stats(kickoff_live, "return_yards"),
        )
    else:
        out.update(
            {
                "v633_kickoff_landing_rows": 0,
                "v633_kickoff_landing_mean": 4.2,
                "v633_kickoff_landing_sd": 3.5,
                "v633_kickoff_landing_p25": 1.0,
                "v633_kickoff_landing_p50": 3.0,
                "v633_kickoff_landing_p75": 6.0,
                "v633_kickoff_returned_drive_start_mean": 29.6,
            }
        )
        _flatten_profile(out, "v633_kickoff_global", _series_stats(pl.DataFrame(), "return_yards"))

    return pl.DataFrame([out])
