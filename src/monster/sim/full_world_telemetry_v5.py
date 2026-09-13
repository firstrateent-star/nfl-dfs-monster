from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from monster.sim.full_world_telemetry_v2 import FullWorldTelemetryV2
from monster.sim.play_kernel import PlayType
from monster.sim.reality_snap_v5 import event_metadata


@dataclass
class FullWorldTelemetryV5(FullWorldTelemetryV2):
    """Extend v2 telemetry with exact Monte Carlo world and 11v11 snap evidence."""

    snap_worlds: list[dict[str, object]] = field(default_factory=list)
    _world_counter: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def capture(self, result: object) -> None:
        final_state = result.final_state
        game = f"{final_state.away_team_id}@{final_state.home_team_id}"
        world = int(self._world_counter[game])
        self._world_counter[game] += 1

        starts = {
            "pass_throws": len(self.pass_throws),
            "dropbacks": len(self.dropbacks),
            "designed_runs": len(self.designed_runs),
            "scrambles": len(self.scrambles),
            "drive_traces": len(self.drive_traces),
        }
        super().capture(result)
        for name, start in starts.items():
            rows = getattr(self, name)
            for row in rows[start:]:
                row["world"] = world

        snap_index = 0
        for event in result.plays:
            if event.play_type not in {PlayType.RUN, PlayType.PASS}:
                continue
            meta = event_metadata(event)
            self.snap_worlds.append(
                {
                    "game": game,
                    "world": world,
                    "snap_index": snap_index,
                    "snap_key": meta.get("snap_key"),
                    "play_type": str(event.play_type),
                    "pass_result": None if event.pass_result is None else str(event.pass_result),
                    "pass_depth": event.pass_depth_category,
                    "run_geometry": event.run_geometry_category,
                    "offense_package": meta.get("offense_package"),
                    "defense_package": meta.get("defense_package"),
                    "coverage_shell": meta.get("coverage_shell"),
                    "rush_plan": meta.get("rush_plan"),
                    "box_aggression": meta.get("box_aggression"),
                    "intent_authority": meta.get("intent_authority"),
                    "passer_id": event.passer_id,
                    "target_id": event.target_id,
                    "rusher_id": event.rusher_id,
                    "primary_defender_id": event.primary_defender_id,
                    "primary_rusher_id": meta.get("primary_rusher_id"),
                    "safety_defender_id": meta.get("safety_defender_id"),
                    "bracket_defender_id": meta.get("bracket_defender_id"),
                    "pursuit_defender_id": meta.get("pursuit_defender_id"),
                    "offense_participants": "|".join(meta.get("offense_participant_ids", ())),
                    "defense_participants": "|".join(meta.get("defense_participant_ids", ())),
                    "offense_alignment": "|".join(meta.get("offense_alignment", ())),
                    "defense_alignment": "|".join(meta.get("defense_alignment", ())),
                    "planned_rushers": "|".join(meta.get("planned_rusher_ids", ())),
                    "run_blockers": "|".join(meta.get("run_blocker_ids", ())),
                    "trench_duels": "|".join(meta.get("trench_duels", ())),
                    "pressured": bool(event.pressured),
                    "stuffed": bool(event.stuffed),
                    "yards": float(event.yards),
                    "air_yards": float(event.air_yards),
                    "yac": float(event.yards_after_catch),
                    "yards_before_contact": float(event.yards_before_contact),
                    "yards_after_contact": float(event.yards_after_contact),
                    "touchdown": bool(event.touchdown),
                    "turnover": bool(event.turnover),
                }
            )
            snap_index += 1

    def write(self, out: Path) -> None:
        super().write(out)
        if not self.snap_worlds:
            return
        snaps = pl.DataFrame(self.snap_worlds)
        snaps.write_csv(out / "same_world_snap_participants_v5.csv")
        snaps.group_by(["offense_package", "defense_package"]).agg(
            pl.len().alias("snaps"),
            pl.col("yards").mean().alias("yards_per_snap"),
            pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
            pl.col("stuffed").cast(pl.Float64).mean().alias("stuff_rate"),
            pl.col("touchdown").cast(pl.Float64).mean().alias("touchdown_rate"),
        ).sort("snaps", descending=True).write_csv(out / "same_world_personnel_package_summary_v5.csv")
        snaps.group_by(["coverage_shell", "rush_plan"]).agg(
            pl.len().alias("snaps"),
            pl.col("yards").mean().alias("yards_per_snap"),
            pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"),
            pl.col("turnover").cast(pl.Float64).mean().alias("turnover_rate"),
            (pl.col("yards") >= 20.0).cast(pl.Float64).mean().alias("explosive_20_rate"),
            (pl.col("yards") >= 40.0).cast(pl.Float64).mean().alias("explosive_40_rate"),
        ).sort("snaps", descending=True).write_csv(out / "same_world_defensive_intent_summary_v5.csv")
