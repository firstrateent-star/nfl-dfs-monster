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
    drive_traces: list[dict[str, object]] = field(default_factory=list)

    def capture(self, result: object) -> None:
        final_state = result.final_state
        game = f"{final_state.away_team_id}@{final_state.home_team_id}"

        for drive_index, trace in enumerate(result.drive_traces):
            terminal = getattr(trace.terminal, "value", str(trace.terminal))
            self.drive_traces.append(
                {
                    "game": game,
                    "drive_index": drive_index,
                    "offense_team_id": trace.offense_team_id,
                    "defense_team_id": trace.defense_team_id,
                    "start_quarter": int(trace.start_quarter),
                    "start_seconds_remaining": int(trace.start_seconds_remaining),
                    "start_yardline_100": float(trace.start_yardline_100),
                    "start_score_margin": int(trace.start_score_margin),
                    "end_quarter": int(trace.end_quarter),
                    "end_seconds_remaining": int(trace.end_seconds_remaining),
                    "end_yardline_100": float(trace.end_yardline_100),
                    "terminal": str(terminal),
                    "points": int(trace.points),
                    "scrimmage_plays": int(trace.scrimmage_plays),
                    "net_scrimmage_yards": float(trace.net_scrimmage_yards),
                    "first_downs": int(trace.first_downs),
                    "explosive_plays": int(trace.explosive_plays),
                    "red_zone_entered": bool(trace.red_zone_entered),
                    "red_zone_snap_seen": bool(trace.red_zone_snap_seen),
                    "goal_to_go_reached": bool(trace.goal_to_go_reached),
                    "goal_to_go_snap_seen": bool(trace.goal_to_go_snap_seen),
                    "pressured_dropbacks": int(trace.pressured_dropbacks),
                    "sacks": int(trace.sacks),
                    "turnovers": int(trace.turnovers),
                    "overtime": bool(trace.overtime),
                    "series_started": int(trace.series_started),
                    "series_converted": int(trace.series_converted),
                    "first_down_snaps": int(trace.first_down_snaps),
                    "second_down_snaps": int(trace.second_down_snaps),
                    "third_down_snaps": int(trace.third_down_snaps),
                    "fourth_down_snaps": int(trace.fourth_down_snaps),
                    "third_down_conversions": int(trace.third_down_conversions),
                    "third_and_long_snaps": int(trace.third_and_long_snaps),
                    "third_and_long_conversions": int(trace.third_and_long_conversions),
                    "third_down_distance_total": float(trace.third_down_distance_total),
                    "early_down_5plus_gains": int(trace.early_down_5plus_gains),
                    "early_down_run_snaps": int(trace.early_down_run_snaps),
                    "early_down_run_yards_total": float(trace.early_down_run_yards_total),
                    "early_down_run_negative_gains": int(trace.early_down_run_negative_gains),
                    "early_down_run_3plus_gains": int(trace.early_down_run_3plus_gains),
                    "early_down_run_5plus_gains": int(trace.early_down_run_5plus_gains),
                    "early_down_run_10plus_gains": int(trace.early_down_run_10plus_gains),
                    "early_down_pass_snaps": int(trace.early_down_pass_snaps),
                    "early_down_pass_yards_total": float(trace.early_down_pass_yards_total),
                    "early_down_pass_negative_gains": int(trace.early_down_pass_negative_gains),
                    "early_down_pass_3plus_gains": int(trace.early_down_pass_3plus_gains),
                    "early_down_pass_5plus_gains": int(trace.early_down_pass_5plus_gains),
                    "early_down_pass_10plus_gains": int(trace.early_down_pass_10plus_gains),
                }
            )

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

    @staticmethod
    def _drive_summary(frame: pl.DataFrame) -> pl.DataFrame:
        series_started = pl.col("series_started").sum()
        third_down_snaps = pl.col("third_down_snaps").sum()
        third_long_snaps = pl.col("third_and_long_snaps").sum()
        early_run_snaps = pl.col("early_down_run_snaps").sum()
        early_pass_snaps = pl.col("early_down_pass_snaps").sum()
        return frame.select(
            pl.len().alias("drives"),
            pl.col("points").mean().alias("points_per_drive"),
            (pl.col("points") > 0).mean().alias("scoring_drive_rate"),
            (pl.col("points") >= 6).mean().alias("touchdown_like_drive_rate"),
            pl.col("scrimmage_plays").mean().alias("plays_per_drive"),
            pl.col("net_scrimmage_yards").mean().alias("net_yards_per_drive"),
            pl.col("first_downs").mean().alias("first_downs_per_drive"),
            (pl.col("explosive_plays") > 0).mean().alias("explosive_drive_rate"),
            pl.col("red_zone_entered").cast(pl.Float64).mean().alias("red_zone_entry_rate"),
            pl.col("red_zone_snap_seen").cast(pl.Float64).mean().alias("red_zone_snap_rate"),
            pl.col("goal_to_go_reached").cast(pl.Float64).mean().alias("goal_to_go_rate"),
            pl.col("series_converted").sum().truediv(series_started.clip(lower_bound=1)).alias(
                "series_conversion_rate"
            ),
            pl.col("third_down_conversions")
            .sum()
            .truediv(third_down_snaps.clip(lower_bound=1))
            .alias("third_down_conversion_rate"),
            pl.col("third_and_long_conversions")
            .sum()
            .truediv(third_long_snaps.clip(lower_bound=1))
            .alias("third_and_long_conversion_rate"),
            pl.col("early_down_run_5plus_gains")
            .sum()
            .truediv(early_run_snaps.clip(lower_bound=1))
            .alias("early_down_run_5plus_rate"),
            pl.col("early_down_pass_5plus_gains")
            .sum()
            .truediv(early_pass_snaps.clip(lower_bound=1))
            .alias("early_down_pass_5plus_rate"),
        )

    def _write_drive_telemetry(self, out: Path) -> None:
        if not self.drive_traces:
            return
        drives = pl.DataFrame(self.drive_traces).with_columns(
            pl.when(pl.col("start_yardline_100") < 20.0)
            .then(pl.lit("own_1_19"))
            .when(pl.col("start_yardline_100") < 40.0)
            .then(pl.lit("own_20_39"))
            .when(pl.col("start_yardline_100") < 60.0)
            .then(pl.lit("own_40_to_plus_41"))
            .when(pl.col("start_yardline_100") < 80.0)
            .then(pl.lit("plus_40_to_21"))
            .otherwise(pl.lit("red_zone"))
            .alias("start_zone"),
            (pl.col("explosive_plays") > 0).alias("had_explosive_play"),
        )
        drives.write_csv(out / "same_world_drive_traces.csv")

        drives.group_by("terminal").agg(
            pl.len().alias("drives"),
            pl.col("points").mean().alias("points_per_drive"),
            pl.col("scrimmage_plays").mean().alias("plays_per_drive"),
            pl.col("net_scrimmage_yards").mean().alias("net_yards_per_drive"),
            pl.col("red_zone_entered").cast(pl.Float64).mean().alias("red_zone_entry_rate"),
            pl.col("had_explosive_play").cast(pl.Float64).mean().alias("explosive_drive_rate"),
        ).sort("drives", descending=True).write_csv(out / "same_world_drive_terminal_summary.csv")

        drives.group_by("start_zone").agg(
            pl.len().alias("drives"),
            pl.col("points").mean().alias("points_per_drive"),
            (pl.col("points") > 0).mean().alias("scoring_drive_rate"),
            (pl.col("points") >= 6).mean().alias("touchdown_like_drive_rate"),
            pl.col("scrimmage_plays").mean().alias("plays_per_drive"),
            pl.col("net_scrimmage_yards").mean().alias("net_yards_per_drive"),
            pl.col("red_zone_entered").cast(pl.Float64).mean().alias("red_zone_entry_rate"),
        ).sort("start_zone").write_csv(out / "same_world_drive_start_zone_summary.csv")

        segments: list[tuple[str, pl.DataFrame]] = [
            ("all_drives", drives),
            ("explosive_drive", drives.filter(pl.col("had_explosive_play"))),
            ("no_explosive_drive", drives.filter(~pl.col("had_explosive_play"))),
            ("red_zone_entered", drives.filter(pl.col("red_zone_entered"))),
            ("red_zone_snap_seen", drives.filter(pl.col("red_zone_snap_seen"))),
            ("goal_to_go_reached", drives.filter(pl.col("goal_to_go_reached"))),
            ("short_field_start", drives.filter(pl.col("start_yardline_100") >= 60.0)),
        ]
        summaries = []
        for segment, frame in segments:
            if frame.is_empty():
                continue
            summaries.append(self._drive_summary(frame).with_columns(pl.lit(segment).alias("segment")))
        if summaries:
            pl.concat(summaries, how="diagonal_relaxed").select(
                "segment", *[column for column in summaries[0].columns if column != "segment"]
            ).write_csv(out / "same_world_drive_progression_summary.csv")

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self._write_drive_telemetry(out)

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
