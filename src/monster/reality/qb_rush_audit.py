from __future__ import annotations

import numpy as np
import polars as pl

QB_RUSH_FAMILIES = (
    "designed_non_sneak",
    "sneak",
    "pressure_scramble",
    "coverage_scramble",
)


def classify_qb_rush_rows(snaps: pl.DataFrame) -> pl.DataFrame:
    """Classify simulated QB rushing from exact snap telemetry.

    QB identity is inferred from same-game/world passer participation. This makes
    the audit robust to backup-QB worlds without requiring a separate starter map.
    """

    required = {
        "game",
        "world",
        "play_type",
        "pass_result",
        "passer_id",
        "rusher_id",
        "run_geometry",
        "pressured",
        "yards",
    }
    missing = required - set(snaps.columns)
    if missing:
        raise ValueError(f"Missing v5 snap telemetry columns: {sorted(missing)}")

    passers = (
        snaps.filter(pl.col("passer_id").is_not_null())
        .select(["game", "world", pl.col("passer_id").alias("qb_id")])
        .unique()
    )
    rows = snaps.join(
        passers,
        left_on=["game", "world", "rusher_id"],
        right_on=["game", "world", "qb_id"],
        how="inner",
    ).with_columns(
        pl.col("play_type").cast(pl.Utf8).str.to_lowercase().alias("_play"),
        pl.col("pass_result")
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .alias("_result"),
        pl.col("run_geometry")
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .alias("_geometry"),
    )

    return (
        rows.with_columns(
            pl.when(pl.col("_result") == "scramble")
            .then(
                pl.when(pl.col("pressured").fill_null(False))
                .then(pl.lit("pressure_scramble"))
                .otherwise(pl.lit("coverage_scramble"))
            )
            .when((pl.col("_play") == "run") & (pl.col("_geometry") == "qb_sneak"))
            .then(pl.lit("sneak"))
            .when(pl.col("_play") == "run")
            .then(pl.lit("designed_non_sneak"))
            .otherwise(pl.lit(None))
            .alias("qb_rush_family")
        )
        .filter(pl.col("qb_rush_family").is_not_null())
    )


def summarize_qb_rush_family_worlds(classified: pl.DataFrame) -> pl.DataFrame:
    if classified.is_empty():
        return pl.DataFrame(
            schema={
                "game": pl.Utf8,
                "world": pl.Int64,
                "qb_id": pl.Utf8,
            }
        )

    grouped = (
        classified.group_by(["game", "world", "rusher_id", "qb_rush_family"])
        .agg(
            pl.len().alias("attempts"),
            pl.col("yards").fill_null(0.0).sum().alias("yards"),
        )
        .rename({"rusher_id": "qb_id"})
        .pivot(
            on="qb_rush_family",
            index=["game", "world", "qb_id"],
            values=["attempts", "yards"],
            aggregate_function="first",
        )
    )

    rename_map: dict[str, str] = {}
    for family in QB_RUSH_FAMILIES:
        for value in ("attempts", "yards"):
            for candidate in (f"{value}_{family}", f"{family}_{value}"):
                if candidate in grouped.columns:
                    rename_map[candidate] = f"{family}_{value}"
    if rename_map:
        grouped = grouped.rename(rename_map)

    for family in QB_RUSH_FAMILIES:
        for value in ("attempts", "yards"):
            name = f"{family}_{value}"
            if name not in grouped.columns:
                grouped = grouped.with_columns(pl.lit(0.0).alias(name))

    grouped = grouped.with_columns(
        *[
            pl.col(f"{family}_{value}").fill_null(0.0).alias(f"{family}_{value}")
            for family in QB_RUSH_FAMILIES
            for value in ("attempts", "yards")
        ]
    ).with_columns(
        (
            pl.col("designed_non_sneak_attempts")
            + pl.col("sneak_attempts")
            + pl.col("pressure_scramble_attempts")
            + pl.col("coverage_scramble_attempts")
        ).alias("competitive_qb_rush_attempts"),
        (
            pl.col("designed_non_sneak_yards")
            + pl.col("sneak_yards")
            + pl.col("pressure_scramble_yards")
            + pl.col("coverage_scramble_yards")
        ).alias("competitive_qb_rush_yards"),
        (
            pl.col("pressure_scramble_attempts")
            + pl.col("coverage_scramble_attempts")
        ).alias("scramble_attempts"),
        (
            pl.col("pressure_scramble_yards")
            + pl.col("coverage_scramble_yards")
        ).alias("scramble_yards"),
    )
    return grouped.sort(["game", "world", "qb_id"])


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def qb_rush_family_manifest(worlds: pl.DataFrame) -> dict[str, object]:
    family_summary: dict[str, dict[str, float]] = {}
    for family in QB_RUSH_FAMILIES:
        attempts = (
            worlds.get_column(f"{family}_attempts").to_numpy()
            if worlds.height
            else np.array([])
        )
        yards = (
            worlds.get_column(f"{family}_yards").to_numpy()
            if worlds.height
            else np.array([])
        )
        family_summary[family] = {
            **{
                f"attempts_{key}": value
                for key, value in distribution_summary(attempts).items()
            },
            **{
                f"yards_{key}": value
                for key, value in distribution_summary(yards).items()
            },
        }

    competitive_attempts = (
        worlds.get_column("competitive_qb_rush_attempts").to_numpy()
        if worlds.height
        else np.array([])
    )
    competitive_yards = (
        worlds.get_column("competitive_qb_rush_yards").to_numpy()
        if worlds.height
        else np.array([])
    )
    scramble = (
        worlds.get_column("scramble_attempts").to_numpy()
        if worlds.height
        else np.array([])
    )
    return {
        "qb_world_rows": int(worlds.height),
        "family_summary": family_summary,
        "competitive_qb_rushing": {
            **{
                f"attempts_{key}": value
                for key, value in distribution_summary(competitive_attempts).items()
            },
            **{
                f"yards_{key}": value
                for key, value in distribution_summary(competitive_yards).items()
            },
        },
        "scramble_attempts": distribution_summary(scramble),
        "classification": {
            "designed_non_sneak": (
                "RUN by a player who is also a passer in the same game-world, "
                "excluding qb_sneak"
            ),
            "sneak": "RUN with run_geometry=qb_sneak by a same-world passer",
            "pressure_scramble": "pass_result=scramble and pressured=true",
            "coverage_scramble": "pass_result=scramble and pressured=false",
            "kneel": (
                "not yet generated as a distinct v6/v7 snap family and therefore "
                "intentionally absent"
            ),
        },
        "market_blind": True,
        "week1_truth_used": False,
        "principle": (
            "Measure QB rushing by causal event family so designed-run authority "
            "cannot silently double-count scramble or kneel history."
        ),
    }
