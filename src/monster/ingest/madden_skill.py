from __future__ import annotations

import re
import unicodedata

import polars as pl

_SKILL = {"QB", "RB", "WR", "TE"}


def _norm(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _first_column(frame: pl.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric(frame: pl.DataFrame, *names: str) -> pl.Expr:
    name = _first_column(frame, *names)
    return pl.col(name).cast(pl.Float64, strict=False) if name else pl.lit(None, dtype=pl.Float64)


def compile_madden_skill_traits(ratings: pl.DataFrame) -> pl.DataFrame:
    """Extract mechanism-level QB/RB/WR/TE traits from Madden 27 ratings.

    Madden is scouting-style proxy evidence. Raw 0-99 attributes are retained so the downstream
    mechanism compiler can shrink/bound their authority; this adapter never creates fantasy points.
    """
    if not ratings.height:
        return pl.DataFrame()
    required = {"full_name", "position"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Madden skill ratings missing columns: {sorted(missing)}")

    frame = ratings.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(sorted(_SKILL)))
    route = pl.mean_horizontal(
        _numeric(frame, "short_route_running_rating", "short_route_rating"),
        _numeric(frame, "medium_route_running_rating", "medium_route_rating"),
        _numeric(frame, "deep_route_running_rating", "deep_route_rating"),
    )
    accuracy = pl.mean_horizontal(
        _numeric(frame, "throw_accuracy_short_rating", "short_throw_accuracy_rating"),
        _numeric(frame, "throw_accuracy_mid_rating", "medium_throw_accuracy_rating"),
        _numeric(frame, "throw_accuracy_deep_rating", "deep_throw_accuracy_rating"),
    )
    return (
        frame.with_columns(
            pl.col("full_name").map_elements(_norm, return_dtype=pl.Utf8).alias("_madden_name_key"),
            pl.col("position").cast(pl.Utf8).str.to_uppercase().alias("_madden_position"),
            _numeric(frame, "speed_rating").alias("madden_speed"),
            _numeric(frame, "acceleration_rating").alias("madden_acceleration"),
            route.alias("madden_route_running"),
            _numeric(frame, "catching_rating").alias("madden_catching"),
            _numeric(frame, "catch_in_traffic_rating").alias("madden_catch_in_traffic"),
            _numeric(frame, "spectacular_catch_rating").alias("madden_spectacular_catch"),
            _numeric(frame, "release_rating").alias("madden_release"),
            _numeric(frame, "carrying_rating").alias("madden_carrying"),
            _numeric(frame, "break_tackle_rating").alias("madden_break_tackle"),
            _numeric(frame, "strength_rating").alias("madden_strength"),
            _numeric(frame, "agility_rating").alias("madden_agility"),
            _numeric(frame, "change_of_direction_rating").alias("madden_change_of_direction"),
            _numeric(frame, "awareness_rating").alias("madden_awareness"),
            _numeric(frame, "throw_power_rating").alias("madden_throw_power"),
            accuracy.alias("madden_throw_accuracy"),
            _numeric(frame, "throw_under_pressure_rating").alias("madden_throw_under_pressure"),
            _numeric(frame, "play_action_rating").alias("madden_play_action"),
            _numeric(frame, "break_sack_rating").alias("madden_break_sack"),
        )
        .select(
            "_madden_name_key", "_madden_position", "madden_speed", "madden_acceleration",
            "madden_route_running", "madden_catching", "madden_catch_in_traffic",
            "madden_spectacular_catch", "madden_release", "madden_carrying",
            "madden_break_tackle", "madden_strength", "madden_agility",
            "madden_change_of_direction", "madden_awareness", "madden_throw_power",
            "madden_throw_accuracy", "madden_throw_under_pressure", "madden_play_action",
            "madden_break_sack",
        )
        .unique(subset=["_madden_name_key", "_madden_position"], keep="last")
    )


def attach_madden_skill_traits(personnel: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Attach current skill-player Madden traits by normalized name + position.

    Team is intentionally not required because offseason transfers can make the Madden roster stale;
    name+position is safer for unique skill identities. Ambiguous duplicate keys are excluded.
    """
    traits = compile_madden_skill_traits(ratings)
    trait_columns = [c for c in traits.columns if c.startswith("madden_") and c not in {"madden_position"}]
    if not traits.height:
        return personnel.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in trait_columns]) if trait_columns else personnel

    counts = traits.group_by(["_madden_name_key", "_madden_position"]).agg(pl.len().alias("_madden_count"))
    traits = traits.join(counts, on=["_madden_name_key", "_madden_position"], how="left").filter(pl.col("_madden_count") == 1).drop("_madden_count")
    name_col = "display_name" if "display_name" in personnel.columns else "full_name"
    current = personnel.with_columns(
        pl.col(name_col).map_elements(_norm, return_dtype=pl.Utf8).alias("_madden_name_key"),
        pl.col("position").cast(pl.Utf8).str.to_uppercase().alias("_madden_position"),
    )
    return current.join(traits, on=["_madden_name_key", "_madden_position"], how="left").drop(["_madden_name_key", "_madden_position"])


def madden_skill_coverage(personnel: pl.DataFrame) -> pl.DataFrame:
    skill = personnel.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(sorted(_SKILL)))
    if not skill.height or "madden_speed" not in skill.columns:
        return pl.DataFrame()
    return skill.group_by("team_id").agg(
        pl.len().alias("skill_roster_rows"),
        pl.col("madden_speed").is_not_null().sum().alias("skill_with_madden_speed"),
        pl.col("madden_acceleration").is_not_null().sum().alias("skill_with_madden_acceleration"),
        pl.col("madden_route_running").is_not_null().sum().alias("skill_with_madden_route"),
        pl.col("madden_catching").is_not_null().sum().alias("skill_with_madden_catching"),
        pl.col("madden_throw_accuracy").is_not_null().sum().alias("qbs_with_madden_accuracy"),
    ).sort("team_id")
