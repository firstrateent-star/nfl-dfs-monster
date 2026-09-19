from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _historical_context() -> pl.Expr:
    qtr = pl.col("qtr").cast(pl.Int64)
    game_left = pl.col("game_seconds_remaining").cast(pl.Float64)
    period = (
        pl.when(qtr == 1).then(game_left - 2700)
        .when(qtr == 2).then(game_left - 1800)
        .when(qtr == 3).then(game_left - 900)
        .otherwise(game_left)
    )
    diff = pl.col("score_differential").cast(pl.Float64)
    return (
        pl.when((qtr == 2) & (period <= 120))
        .then(pl.lit("q2_two_minute"))
        .when((qtr == 4) & (period <= 240) & (diff < 0))
        .then(pl.lit("q4_trailing"))
        .when((qtr == 4) & (period <= 240) & (diff >= 7))
        .then(pl.lit("q4_lead_drain"))
        .otherwise(pl.lit("ordinary"))
    )


def _sim_context() -> pl.Expr:
    qtr = pl.col("quarter")
    clock = pl.col("period_clock")
    margin = pl.col("margin")
    return (
        pl.when((qtr == 2) & (clock <= 120))
        .then(pl.lit("q2_two_minute"))
        .when((qtr == 4) & (clock <= 240) & (margin < 0))
        .then(pl.lit("q4_trailing"))
        .when((qtr == 4) & (clock <= 240) & (margin >= 7))
        .then(pl.lit("q4_lead_drain"))
        .otherwise(pl.lit("ordinary"))
    )


def historical_clock_ecology(season: int, cache: Path) -> pl.DataFrame:
    configure_cache(cache)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    defaults = {
        "qb_dropback": 0,
        "rush_attempt": 0,
        "complete_pass": 0,
        "sack": 0,
        "qb_scramble": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "out_of_bounds": 0,
        "score_differential": 0.0,
    }
    missing = [
        pl.lit(value).alias(name)
        for name, value in defaults.items()
        if name not in pbp.columns
    ]
    if missing:
        pbp = pbp.with_columns(*missing)

    required = {
        "game_id",
        "fixed_drive",
        "play_id",
        "posteam",
        "qtr",
        "game_seconds_remaining",
    }
    absent = required.difference(pbp.columns)
    if absent:
        raise ValueError(f"historical clock audit missing fields: {sorted(absent)}")

    dropback = pl.col("qb_dropback").fill_null(0).cast(pl.Int64) == 1
    designed_run = (
        pl.col("rush_attempt").fill_null(0).cast(pl.Int64) == 1
    ) & ~dropback
    live_result = (
        designed_run
        | (pl.col("complete_pass").fill_null(0).cast(pl.Int64) == 1)
        | (pl.col("sack").fill_null(0).cast(pl.Int64) == 1)
        | (pl.col("qb_scramble").fill_null(0).cast(pl.Int64) == 1)
    )

    frame = (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & pl.col("game_seconds_remaining").is_not_null()
            & (pl.col("qb_kneel").fill_null(0) != 1)
            & (pl.col("qb_spike").fill_null(0) != 1)
            & live_result
            & (pl.col("out_of_bounds").fill_null(0) != 1)
        )
        .sort(["game_id", "fixed_drive", "play_id"])
        .with_columns(
            pl.col("game_seconds_remaining")
            .shift(-1)
            .over(["game_id", "fixed_drive"])
            .alias("next_game_seconds_remaining"),
            pl.col("qtr")
            .shift(-1)
            .over(["game_id", "fixed_drive"])
            .alias("next_qtr"),
            _historical_context().alias("context"),
        )
        .with_columns(
            (
                pl.col("game_seconds_remaining")
                - pl.col("next_game_seconds_remaining")
            ).alias("elapsed")
        )
        .filter(
            pl.col("next_game_seconds_remaining").is_not_null()
            & (pl.col("next_qtr") == pl.col("qtr"))
            & pl.col("elapsed").is_between(0, 60)
        )
    )

    return (
        frame.group_by("context")
        .agg(
            pl.len().alias("samples"),
            pl.col("elapsed").mean().alias("historical_elapsed_mean"),
            pl.col("elapsed").median().alias("historical_elapsed_p50"),
            (pl.col("elapsed") <= 10).mean().alias("historical_fast_stop_rate"),
        )
        .sort("context")
    )


def simulated_clock_ecology(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    live_clock = (
        (pl.col("play_type") == "run")
        | (
            (pl.col("play_type") == "pass")
            & pl.col("pass_result").is_in(["complete", "sack", "scramble"])
        )
    )
    frame = frame.filter(live_clock).with_columns(_sim_context().alias("context"))
    return (
        frame.group_by("context")
        .agg(
            pl.len().alias("samples"),
            pl.col("elapsed_after_clock_management").mean().alias(
                "simulated_elapsed_mean"
            ),
            pl.col("elapsed_before_clock_management").mean().alias(
                "pre_v721_elapsed_mean"
            ),
            (pl.col("timeout_team") != "").mean().alias("simulated_timeout_rate"),
            pl.col("hurry_probability").mean().alias("simulated_hurry_mean"),
        )
        .sort("context")
    )


def _value(frame: pl.DataFrame, context: str, column: str) -> float | None:
    hit = frame.filter(pl.col("context") == context)
    if hit.height == 0:
        return None
    return float(hit[column][0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache", type=Path, default=Path(".cache/nflverse"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hist = historical_clock_ecology(args.season, args.cache)
    sim = simulated_clock_ecology(args.simulated)
    joined = hist.join(sim, on="context", how="outer_coalesce").sort("context")
    joined = joined.with_columns(
        (
            pl.col("simulated_elapsed_mean") - pl.col("historical_elapsed_mean")
        )
        .abs()
        .alias("elapsed_abs_error")
    )

    ordinary = _value(sim, "ordinary", "simulated_elapsed_mean")
    q2 = _value(sim, "q2_two_minute", "simulated_elapsed_mean")
    trailing = _value(sim, "q4_trailing", "simulated_elapsed_mean")
    lead = _value(sim, "q4_lead_drain", "simulated_elapsed_mean")

    hist_ordinary = _value(hist, "ordinary", "historical_elapsed_mean")
    hist_q2 = _value(hist, "q2_two_minute", "historical_elapsed_mean")
    hist_trailing = _value(hist, "q4_trailing", "historical_elapsed_mean")
    hist_lead = _value(hist, "q4_lead_drain", "historical_elapsed_mean")

    timeout_rows = pl.read_csv(args.simulated).filter(pl.col("timeout_team") != "")
    timeout_worlds = (
        timeout_rows.select(["game", "world_key"]).unique().height
        if timeout_rows.height
        else 0
    )

    comparable = joined.drop_nulls(
        ["simulated_elapsed_mean", "historical_elapsed_mean"]
    )
    mean_abs_error = (
        float(comparable["elapsed_abs_error"].mean())
        if comparable.height
        else float("nan")
    )

    manifest = {
        "artifact": "MONSTER V7.2.1 Game Script / Clock Reality Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "historical_season": args.season,
        "market_blind": True,
        "final_scores_used_as_targets": False,
        "fantasy_points_used_as_targets": False,
        "contexts_compared": int(comparable.height),
        "mean_context_elapsed_abs_error_seconds": mean_abs_error,
        "sim_timeout_events": int(timeout_rows.height),
        "sim_worlds_with_timeout": int(timeout_worlds),
        "sim_q2_two_minute_faster_than_ordinary": bool(
            ordinary is not None and q2 is not None and q2 < ordinary
        ),
        "sim_q4_trailing_faster_than_ordinary": bool(
            ordinary is not None and trailing is not None and trailing < ordinary
        ),
        "sim_q4_lead_slower_than_trailing": bool(
            lead is not None and trailing is not None and lead > trailing
        ),
        "historical_q2_two_minute_faster_than_ordinary": bool(
            hist_ordinary is not None and hist_q2 is not None and hist_q2 < hist_ordinary
        ),
        "historical_q4_trailing_faster_than_ordinary": bool(
            hist_ordinary is not None
            and hist_trailing is not None
            and hist_trailing < hist_ordinary
        ),
        "historical_q4_lead_slower_than_trailing": bool(
            hist_lead is not None
            and hist_trailing is not None
            and hist_lead > hist_trailing
        ),
        "principle": (
            "Clock/script mechanisms are judged on state-conditioned football behavior, "
            "not on matching game scores or DFS outputs."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    hist.write_csv(args.out / "historical_clock_context.csv")
    sim.write_csv(args.out / "simulated_clock_context.csv")
    joined.write_csv(args.out / "clock_context_comparison.csv")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(joined)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
