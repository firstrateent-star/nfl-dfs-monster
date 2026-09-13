from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from monster.dfs.defense import score_defense_worlds
from monster.sim.chaos_ecology import ReturnKind
from monster.sim.football_state import PossessionTerminal


@dataclass
class SameWorldDSTCollector:
    """Materialize FanDuel D/ST from the exact Monster game worlds used by offense."""

    rows: list[dict[str, object]] = field(default_factory=list)
    _world_counter: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def capture(self, result: object) -> None:
        state = result.final_state
        away = str(state.away_team_id)
        home = str(state.home_team_id)
        game = f"{away}@{home}"
        world = int(self._world_counter[game])
        self._world_counter[game] += 1

        metrics = {
            away: {
                "sacks": 0,
                "turnovers": 0,
                "defensive_tds": 0,
                "special_teams_tds": 0,
                "safeties": 0,
                "blocked_kicks": 0,
            },
            home: {
                "sacks": 0,
                "turnovers": 0,
                "defensive_tds": 0,
                "special_teams_tds": 0,
                "safeties": 0,
                "blocked_kicks": 0,
            },
        }

        # Drive traces retain exact offense/defense ownership, so sacks and safeties can be
        # attributed without reconstructing pre-snap state from a flattened play ledger.
        for drive in result.drive_traces:
            defense_team = str(drive.defense_team_id)
            if defense_team not in metrics:
                continue
            metrics[defense_team]["sacks"] += int(drive.sacks)
            metrics[defense_team]["safeties"] += int(drive.terminal == PossessionTerminal.SAFETY)

        for ret in result.return_events:
            return_team = str(ret.return_team_id)
            if return_team not in metrics:
                continue
            if ret.kind in {ReturnKind.INTERCEPTION, ReturnKind.FUMBLE}:
                metrics[return_team]["turnovers"] += 1
                metrics[return_team]["defensive_tds"] += int(ret.touchdown)
            elif ret.kind in {ReturnKind.PUNT, ReturnKind.KICKOFF}:
                # A muff recovered by the kicking team is an opponent fumble recovery for D/ST.
                metrics[return_team]["turnovers"] += int(ret.muffed and ret.kicking_team_recovery)
                metrics[return_team]["special_teams_tds"] += int(ret.touchdown)
            elif ret.kind in {ReturnKind.BLOCKED_PUNT, ReturnKind.BLOCKED_FIELD_GOAL}:
                metrics[return_team]["blocked_kicks"] += 1
                metrics[return_team]["special_teams_tds"] += int(ret.touchdown)

        final_points = {away: int(state.away_score), home: int(state.home_score)}
        for team, opponent in ((away, home), (home, away)):
            row = metrics[team]
            points_allowed = final_points[opponent]
            score = float(
                score_defense_worlds(
                    opponent_points=np.asarray([points_allowed]),
                    opponent_turnovers=np.asarray([row["turnovers"]]),
                    sacks=np.asarray([row["sacks"]]),
                    defensive_touchdowns=np.asarray([row["defensive_tds"]]),
                    special_teams_touchdowns=np.asarray([row["special_teams_tds"]]),
                    safeties=np.asarray([row["safeties"]]),
                    blocked_kicks=np.asarray([row["blocked_kicks"]]),
                )[0]
            )
            self.rows.append(
                {
                    "game": game,
                    "world": world,
                    "team": team,
                    "opponent": opponent,
                    "player_id": f"DST_{team}",
                    "player": f"{team} D/ST",
                    "position": "D",
                    "fanduel_points": score,
                    "points_allowed": points_allowed,
                    **row,
                }
            )

    def write(self, out: Path) -> None:
        if not self.rows:
            return
        out.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(self.rows).sort(["game", "world", "team"])
        frame.write_csv(out / "dst_world_fanduel.csv")
        frame.group_by(["team", "game"]).agg(
            pl.len().alias("worlds"),
            pl.col("fanduel_points").mean().alias("fanduel_mean"),
            pl.col("fanduel_points").quantile(0.90).alias("fanduel_p90"),
            pl.col("fanduel_points").quantile(0.99).alias("fanduel_p99"),
            pl.col("sacks").mean().alias("sacks_mean"),
            pl.col("turnovers").mean().alias("turnovers_mean"),
            (pl.col("defensive_tds") + pl.col("special_teams_tds")).mean().alias("dst_tds_mean"),
            pl.col("blocked_kicks").mean().alias("blocked_kicks_mean"),
            pl.col("points_allowed").mean().alias("points_allowed_mean"),
        ).sort("fanduel_mean", descending=True).write_csv(out / "dst_distributions_fanduel.csv")


def write_combined_fanduel_worlds(out: Path) -> None:
    offense_path = out / "player_world_fanduel.csv"
    defense_path = out / "dst_world_fanduel.csv"
    if not offense_path.exists() or not defense_path.exists():
        return
    offense = pl.read_csv(offense_path).select(
        "game", "world", "player_id", "player", "position", "fanduel_points"
    )
    defense = pl.read_csv(defense_path).select(
        "game", "world", "player_id", "player", "position", "fanduel_points"
    )
    pl.concat([offense, defense], how="vertical").sort(
        ["game", "world", "position", "player_id"]
    ).write_csv(out / "dfs_world_fanduel_complete.csv")
