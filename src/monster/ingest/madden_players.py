from __future__ import annotations

from io import BytesIO

import polars as pl
import requests

from monster.ingest.madden_schema import normalize_madden_name, normalize_madden_team

MADDEN27_RATINGS_URL = (
    "https://raw.githubusercontent.com/zachxwalton/madden-ratings-breakdown/"
    "main/scraper/output/madden27_ratings.csv"
)
_OL_POSITIONS = {"LT", "LG", "C", "RG", "RT", "T", "G", "OL"}


def load_madden27_player_ratings(url: str = MADDEN27_RATINGS_URL, *, timeout: int = 60) -> pl.DataFrame:
    """Load the public fallback extraction of EA Madden 27 ratings."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pl.read_csv(BytesIO(response.content), infer_schema_length=3000)


def _normalize_name_text(value: str | None) -> str:
    return normalize_madden_name(value)


def _normalize_team_id(value: str | None) -> str | None:
    return normalize_madden_team(value)


def _normalize_names(frame: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(column).cast(pl.Utf8).map_elements(_normalize_name_text, return_dtype=pl.Utf8).alias(alias)
    )


def _low_authority_rating(composite: pl.Expr, center: float = 78.0, weight: float = 0.35) -> pl.Expr:
    return (center + weight * (composite - center)).clip(55.0, 95.0)


def _legacy_ol_schema(ratings: pl.DataFrame) -> pl.DataFrame:
    identity_aliases = {
        "full_name": "madden_player_name",
        "position": "madden_position",
        "team_name": "madden_team",
    }
    numeric_aliases = {
        "pass_block_rating": "madden_pass_block",
        "pass_block_power_rating": "madden_pass_block_power",
        "pass_block_finesse_rating": "madden_pass_block_finesse",
        "run_block_rating": "madden_run_block",
        "run_block_power_rating": "madden_run_block_power",
        "run_block_finesse_rating": "madden_run_block_finesse",
        "impact_block_rating": "madden_impact_blocking",
        "awareness_rating": "madden_awareness",
        "strength_rating": "madden_strength",
        "injury_rating": "madden_injury",
        "stamina_rating": "madden_stamina",
    }
    expressions: list[pl.Expr] = []
    for target, source in identity_aliases.items():
        if target not in ratings.columns and source in ratings.columns:
            expressions.append(pl.col(source).alias(target))
    for target, source in numeric_aliases.items():
        if target in ratings.columns:
            continue
        if source in ratings.columns:
            expressions.append(pl.col(source).alias(target))
        else:
            expressions.append(pl.lit(78.0, dtype=pl.Float64).alias(target))
    return ratings.with_columns(*expressions) if expressions else ratings


def compile_madden_ol_proxies(ratings: pl.DataFrame) -> pl.DataFrame:
    ratings = _legacy_ol_schema(ratings)
    required = {
        "full_name", "position", "team_name", "pass_block_rating",
        "pass_block_power_rating", "pass_block_finesse_rating", "run_block_rating",
        "run_block_power_rating", "run_block_finesse_rating", "impact_block_rating",
        "awareness_rating", "strength_rating", "injury_rating", "stamina_rating",
    }
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Madden OL ratings missing identity columns: {sorted(missing)}")
    ol = ratings.filter(pl.col("position").cast(pl.Utf8).is_in(sorted(_OL_POSITIONS)))
    ol = ol.with_columns(
        pl.col("team_name").cast(pl.Utf8).map_elements(_normalize_team_id, return_dtype=pl.Utf8).alias("madden_team_id"),
        *[
            pl.col(c).cast(pl.Float64, strict=False)
            for c in [
                "pass_block_rating", "pass_block_power_rating", "pass_block_finesse_rating",
                "run_block_rating", "run_block_power_rating", "run_block_finesse_rating",
                "impact_block_rating", "awareness_rating", "strength_rating", "injury_rating", "stamina_rating",
            ]
        ],
    )
    ol = _normalize_names(ol, "full_name", "_name_key")
    pass_composite = (
        0.45 * pl.col("pass_block_rating") + 0.20 * pl.col("pass_block_power_rating")
        + 0.20 * pl.col("pass_block_finesse_rating") + 0.10 * pl.col("awareness_rating")
        + 0.05 * pl.col("strength_rating")
    )
    run_composite = (
        0.40 * pl.col("run_block_rating") + 0.20 * pl.col("run_block_power_rating")
        + 0.20 * pl.col("run_block_finesse_rating") + 0.10 * pl.col("impact_block_rating")
        + 0.05 * pl.col("strength_rating") + 0.05 * pl.col("awareness_rating")
    )
    return ol.with_columns(
        pass_composite.alias("madden_pass_block_composite_raw"),
        run_composite.alias("madden_run_block_composite_raw"),
        _low_authority_rating(pass_composite).alias("madden_pass_block"),
        _low_authority_rating(run_composite).alias("madden_run_block"),
        (0.55 * pl.col("injury_rating") + 0.45 * pl.col("stamina_rating")).alias("madden_ol_durability_proxy"),
    ).select(
        "_name_key", "full_name", "position", "madden_team_id",
        "madden_pass_block_composite_raw", "madden_run_block_composite_raw",
        "madden_pass_block", "madden_run_block", "madden_ol_durability_proxy",
        "awareness_rating", "strength_rating",
    )


def attach_madden_ol_ratings(personnel: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Attach OL proxy evidence without colliding with canonical official EA attributes."""
    if not ratings.height:
        additions = []
        for column, dtype in (
            ("madden_pass_block", pl.Float64), ("madden_run_block", pl.Float64),
            ("madden_ol_durability_proxy", pl.Float64), ("madden_match_type", pl.Utf8),
        ):
            if column not in personnel.columns:
                additions.append(pl.lit(None, dtype=dtype).alias(column))
        return personnel.with_columns(additions) if additions else personnel

    proxies = compile_madden_ol_proxies(ratings)
    name_column = "display_name" if "display_name" in personnel.columns else "full_name"
    current = _normalize_names(personnel, name_column, "_name_key")
    exact = proxies.select(
        "_name_key",
        pl.col("madden_team_id").alias("team_id"),
        pl.col("madden_pass_block").alias("_team_pass_block"),
        pl.col("madden_run_block").alias("_team_run_block"),
        pl.col("madden_ol_durability_proxy").alias("_team_durability"),
    ).unique(subset=["_name_key", "team_id"], keep="last")
    out = current.join(exact, on=["_name_key", "team_id"], how="left")
    unique_names = (
        proxies.group_by("_name_key")
        .agg(
            pl.len().alias("_madden_name_count"),
            pl.col("madden_pass_block").first().alias("_name_pass_block"),
            pl.col("madden_run_block").first().alias("_name_run_block"),
            pl.col("madden_ol_durability_proxy").first().alias("_name_durability"),
        )
        .filter(pl.col("_madden_name_count") == 1)
    )
    out = out.join(unique_names, on="_name_key", how="left")
    is_ol = pl.col("position_group") == "OL" if "position_group" in out.columns else pl.lit(False)
    existing_pass = pl.col("madden_pass_block") if "madden_pass_block" in out.columns else pl.lit(None, dtype=pl.Float64)
    existing_run = pl.col("madden_run_block") if "madden_run_block" in out.columns else pl.lit(None, dtype=pl.Float64)
    existing_durability = pl.col("madden_ol_durability_proxy") if "madden_ol_durability_proxy" in out.columns else pl.lit(None, dtype=pl.Float64)
    out = out.with_columns(
        pl.when(is_ol).then(pl.coalesce([existing_pass, pl.col("_team_pass_block"), pl.col("_name_pass_block")])).otherwise(existing_pass).alias("madden_pass_block"),
        pl.when(is_ol).then(pl.coalesce([existing_run, pl.col("_team_run_block"), pl.col("_name_run_block")])).otherwise(existing_run).alias("madden_run_block"),
        pl.when(is_ol).then(pl.coalesce([existing_durability, pl.col("_team_durability"), pl.col("_name_durability")])).otherwise(existing_durability).alias("madden_ol_durability_proxy"),
        pl.when(~is_ol).then(None)
        .when(pl.col("_team_pass_block").is_not_null()).then(pl.lit("team_name"))
        .when(pl.col("_name_pass_block").is_not_null()).then(pl.lit("unique_name"))
        .otherwise(None).alias("madden_match_type"),
    )
    return out.drop([
        "_name_key", "_team_pass_block", "_team_run_block", "_team_durability",
        "_madden_name_count", "_name_pass_block", "_name_run_block", "_name_durability",
    ])


def madden_ol_coverage(personnel: pl.DataFrame) -> pl.DataFrame:
    if "madden_pass_block" not in personnel.columns:
        return pl.DataFrame()
    return (
        personnel.filter(pl.col("position_group") == "OL")
        .group_by("team_id")
        .agg(
            pl.len().alias("ol_roster_rows"),
            pl.col("madden_pass_block").is_not_null().sum().alias("ol_with_madden_blocking"),
            (pl.col("madden_match_type") == "team_name").sum().alias("madden_team_name_matches"),
            (pl.col("madden_match_type") == "unique_name").sum().alias("madden_unique_name_matches"),
        )
        .sort("team_id")
    )
