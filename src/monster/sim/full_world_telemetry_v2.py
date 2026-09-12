from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from monster.sim.play_kernel import PassResult, PlayType


@dataclass
class FullWorldTelemetryV2:
    """Collect topology from the exact games that feed scoreboard, stats and DFS outputs."""

    pass_throws: list[dict[str, object]] = field(default_factory=list)
    dropbacks: list[dict[str, object]] = field(default_factory=list)
    designed_runs: list[dict[str, object]] = field(default_factory=list)
    scrambles: list[dict[str, object]] = field(default_factory=list)

    def capture(self, result: object) -> None:
        final_state = result.final_state
        game = f"{final_state.away_team_id}@{final_state.home_team_id}"
        for event in result.plays:
            if event.play_type == PlayType.RUN:
                self.designed_runs.append(
                    {
                        "game": game,
                        "category": event.run_geometry_category or "other",
                        "yards": float(event.yards),
                        "touchdown": bool(event.touchdown),
                        "stuffed": bool(event.stuffed),
                    }
                )
                continue
            if event.play_type != PlayType.PASS:
                continue

            outcome = str(event.pass_result) if event.pass_result is not None else "unknown"
            self.dropbacks.append(
                {
                    "game": game,
                    "outcome": outcome,
                    "category": event.pass_depth_category or "none",
                    "yards": float(event.yards),
                    "pressured": bool(event.pressured),
                    "touchdown": bool(event.touchdown),
                }
            )
            if event.pass_result == PassResult.SCRAMBLE:
                self.scrambles.append(
                    {
                        "game": game,
                        "yards": float(event.yards),
                        "pressured": bool(event.pressured),
                        "touchdown": bool(event.touchdown),
                    }
                )
            if event.pass_result not in {
                PassResult.COMPLETE,
                PassResult.INCOMPLETE,
                PassResult.INTERCEPTION,
            }:
                continue
            self.pass_throws.append(
                {
                    "game": game,
                    "category": event.pass_depth_category or "none",
                    "outcome": outcome,
                    "complete": event.pass_result == PassResult.COMPLETE,
                    "interception": event.pass_result == PassResult.INTERCEPTION,
                    "pressured": bool(event.pressured),
                    "yards": float(event.yards),
                    "air_yards": float(event.air_yards),
                    "yac": float(event.yards_after_catch),
                    "touchdown": bool(event.touchdown),
                }
            )

    @staticmethod
    def _gain_aggregations() -> list[pl.Expr]:
        return [
            pl.col("yards").mean().alias("yards_mean"),
            (pl.col("yards") < 0.0).mean().alias("negative_rate"),
            (pl.col("yards") >= 3.0).mean().alias("gain_3plus_rate"),
            (pl.col("yards") >= 5.0).mean().alias("gain_5plus_rate"),
            (pl.col("yards") >= 10.0).mean().alias("gain_10plus_rate"),
            (pl.col("yards") >= 15.0).mean().alias("gain_15plus_rate"),
            (pl.col("yards") >= 20.0).mean().alias("gain_20plus_rate"),
        ]

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)

        if self.pass_throws:
            throws = pl.DataFrame(self.pass_throws)
            throws.write_csv(out / "same_world_pass_throws.csv")
            total = throws.height
            by_depth = (
                throws.group_by("category")
                .agg(
                    pl.len().alias("attempts"),
                    pl.col("complete").cast(pl.Float64).mean().alias("completion_rate"),
                    pl.col("interception").cast(pl.Float64).mean().alias("interception_rate"),
                    pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
                    pl.col("air_yards").mean().alias("air_yards_mean"),
                    pl.col("yac").filter(pl.col("complete")).mean().fill_null(0.0).alias("yac_mean_complete"),
                    *self._gain_aggregations(),
                )
                .with_columns((pl.col("attempts") / total).alias("share"))
                .sort("category")
            )
            by_depth.write_csv(out / "same_world_pass_depth_summary.csv")
            throws.select(
                pl.len().alias("attempts"),
                pl.col("complete").cast(pl.Float64).mean().alias("completion_rate"),
                pl.col("interception").cast(pl.Float64).mean().alias("interception_rate"),
                pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
                pl.col("air_yards").mean().alias("air_yards_mean"),
                pl.col("yac").filter(pl.col("complete")).mean().fill_null(0.0).alias("yac_mean_complete"),
                *self._gain_aggregations(),
            ).write_csv(out / "same_world_pass_throw_summary.csv")

        if self.dropbacks:
            dropbacks = pl.DataFrame(self.dropbacks)
            dropbacks.write_csv(out / "same_world_dropbacks.csv")
            total = dropbacks.height
            (
                dropbacks.group_by("outcome")
                .agg(
                    pl.len().alias("events"),
                    pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
                    *self._gain_aggregations(),
                )
                .with_columns((pl.col("events") / total).alias("share"))
                .sort("outcome")
                .write_csv(out / "same_world_dropback_outcomes.csv")
            )

        if self.designed_runs:
            runs = pl.DataFrame(self.designed_runs)
            runs.write_csv(out / "same_world_designed_runs.csv")
            total = runs.height
            (
                runs.group_by("category")
                .agg(
                    pl.len().alias("attempts"),
                    pl.col("stuffed").cast(pl.Float64).mean().alias("stuffed_rate"),
                    *self._gain_aggregations(),
                )
                .with_columns((pl.col("attempts") / total).alias("share"))
                .sort("category")
                .write_csv(out / "same_world_run_geometry_summary.csv")
            )

        if self.scrambles:
            scrambles = pl.DataFrame(self.scrambles)
            scrambles.write_csv(out / "same_world_scrambles.csv")
            scrambles.select(
                pl.len().alias("attempts"),
                pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
                *self._gain_aggregations(),
            ).write_csv(out / "same_world_scramble_summary.csv")
