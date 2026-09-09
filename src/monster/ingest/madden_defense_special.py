from __future__ import annotations

import re
import unicodedata

import polars as pl


def _norm(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _first(frame: pl.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _num(frame: pl.DataFrame, *names: str) -> pl.Expr:
    name = _first(frame, *names)
    return pl.col(name).cast(pl.Float64, strict=False) if name else pl.lit(None, dtype=pl.Float64)


def compile_madden_defense_special_traits(ratings: pl.DataFrame) -> pl.DataFrame:
    """Compile defensive and special-teams Madden evidence into mechanism-level proxies."""
    if not ratings.height:
        return pl.DataFrame()
    if "full_name" not in ratings.columns:
        raise ValueError("Madden ratings missing full_name")

    pass_rush = pl.mean_horizontal(
        _num(ratings, "power_moves_rating"),
        _num(ratings, "finesse_moves_rating"),
        _num(ratings, "block_shedding_rating"),
        _num(ratings, "pursuit_rating"),
    )
    coverage = pl.mean_horizontal(
        _num(ratings, "man_coverage_rating"),
        _num(ratings, "zone_coverage_rating"),
        _num(ratings, "press_rating"),
        _num(ratings, "play_recognition_rating"),
    )
    tackle = pl.mean_horizontal(
        _num(ratings, "tackle_rating"),
        _num(ratings, "pursuit_rating"),
        _num(ratings, "hit_power_rating"),
        _num(ratings, "play_recognition_rating"),
    )
    return ratings.with_columns(
        pl.col("full_name").map_elements(_norm, return_dtype=pl.Utf8).alias("_madden_ds_name"),
        pass_rush.alias("_ds_pass_rush"),
        coverage.alias("_ds_coverage"),
        tackle.alias("_ds_tackle"),
        _num(ratings, "speed_rating").alias("_ds_speed"),
        _num(ratings, "accel_rating", "acceleration_rating").alias("_ds_acceleration"),
        _num(ratings, "awareness_rating").alias("_ds_awareness"),
        _num(ratings, "kick_power_rating").alias("_ds_kick_power"),
        _num(ratings, "kick_accuracy_rating").alias("_ds_kick_accuracy"),
        _num(ratings, "kick_return_rating").alias("_ds_return"),
    ).select(
        "_madden_ds_name",
        "_ds_pass_rush",
        "_ds_coverage",
        "_ds_tackle",
        "_ds_speed",
        "_ds_acceleration",
        "_ds_awareness",
        "_ds_kick_power",
        "_ds_kick_accuracy",
        "_ds_return",
    ).unique(subset=["_madden_ds_name"], keep="none")


def attach_madden_defense_special_traits(personnel: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Attach unique-name defensive/ST proxy evidence without overriding existing skill traits."""
    traits = compile_madden_defense_special_traits(ratings)
    if not traits.height:
        return personnel
    name_col = "display_name" if "display_name" in personnel.columns else "full_name"
    out = personnel.with_columns(
        pl.col(name_col).map_elements(_norm, return_dtype=pl.Utf8).alias("_madden_ds_name")
    ).join(traits, on="_madden_ds_name", how="left")

    mapping = {
        "madden_pass_rush": "_ds_pass_rush",
        "madden_coverage": "_ds_coverage",
        "madden_tackle": "_ds_tackle",
        "madden_speed": "_ds_speed",
        "madden_acceleration": "_ds_acceleration",
        "madden_awareness": "_ds_awareness",
        "madden_kick_power": "_ds_kick_power",
        "madden_kick_accuracy": "_ds_kick_accuracy",
        "madden_return": "_ds_return",
    }
    expressions = []
    for target, source in mapping.items():
        if target in out.columns:
            expressions.append(pl.coalesce([pl.col(target), pl.col(source)]).alias(target))
        else:
            expressions.append(pl.col(source).alias(target))
    return out.with_columns(expressions).drop(["_madden_ds_name", *mapping.values()])


def madden_defense_special_coverage(personnel: pl.DataFrame) -> dict[str, int]:
    def count(column: str) -> int:
        return int(personnel.select(pl.col(column).is_not_null().sum()).item()) if column in personnel.columns else 0

    return {
        "pass_rush": count("madden_pass_rush"),
        "coverage": count("madden_coverage"),
        "tackle": count("madden_tackle"),
        "kick_power": count("madden_kick_power"),
        "kick_accuracy": count("madden_kick_accuracy"),
        "return": count("madden_return"),
    }
