from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass
class PlayerSkillTelemetryV2:
    """Persist the player/defender capability state actually handed to the event simulator.

    This is deliberately separate from play outcome telemetry.  It lets audits join a player's
    opportunity and rich identity to 15+/20+/40+ outcomes without copying Madden ratings into
    every play row or giving telemetry any authority over simulation behavior.
    """

    players: dict[str, dict[str, object]] = field(default_factory=dict)
    defenders: dict[str, dict[str, object]] = field(default_factory=dict)

    @staticmethod
    def _channel(player: object, name: str) -> float:
        return float(getattr(player, f"{name}_skill", 0.0) or 0.0)

    def register_team_identity(self, team: object) -> None:
        team_id = str(getattr(team, "team_id", ""))
        candidates = [getattr(team, "quarterback", None)]
        candidates.extend(tuple(getattr(team, "rushers", ())))
        candidates.extend(tuple(getattr(team, "receivers", ())))
        for player in candidates:
            if player is None:
                continue
            player_id = str(getattr(player, "player_id", ""))
            if not player_id:
                continue
            usage = float(getattr(player, "usage_weight", 0.0) or 0.0)
            current = self.players.get(player_id)
            row = {
                "player_id": player_id,
                "player_name": str(getattr(player, "name", player_id)),
                "position": str(getattr(player, "position", "")),
                "team": team_id,
                "usage_weight": usage,
                "efficiency": float(getattr(player, "efficiency", 1.0) or 1.0),
                "explosive": float(getattr(player, "explosive", 1.0) or 1.0),
                "turnover_security": float(
                    getattr(player, "turnover_security", 1.0) or 1.0
                ),
                "speed_skill": self._channel(player, "speed"),
                "mobility_skill": self._channel(player, "mobility"),
                "qb_execution_skill": self._channel(player, "qb_execution"),
                "route_separation_skill": self._channel(player, "route_separation"),
                "catchpoint_skill": self._channel(player, "catchpoint"),
                "rush_creation_skill": self._channel(player, "rush_creation"),
                "runner_power_skill": self._channel(player, "runner_power"),
                "open_field_skill": self._channel(player, "open_field"),
                "ball_security_skill": self._channel(player, "ball_security"),
                "evidence_fields": int(getattr(player, "evidence_fields", 0) or 0),
            }
            # A player can appear in both the rushing and receiving rotations. Preserve the
            # strongest opportunity weight while keeping the same immutable capability state.
            if current is not None and float(current.get("usage_weight", 0.0)) > usage:
                row["usage_weight"] = current["usage_weight"]
            self.players[player_id] = row

    def register_defensive_unit(self, defense: object) -> None:
        seen: set[str] = set()
        for defender in tuple(getattr(defense, "front", ())) + tuple(
            getattr(defense, "coverage", ())
        ):
            player_id = str(getattr(defender, "player_id", ""))
            if not player_id or player_id in seen:
                continue
            seen.add(player_id)
            self.defenders[player_id] = {
                "player_id": player_id,
                "position": str(getattr(defender, "position", "")),
                "coverage": float(getattr(defender, "coverage", 1.0) or 1.0),
                "tackling": float(getattr(defender, "tackling", 1.0) or 1.0),
                "run_defense": float(getattr(defender, "run_defense", 1.0) or 1.0),
                "pass_rush": float(getattr(defender, "pass_rush", 1.0) or 1.0),
                "speed": float(getattr(defender, "speed", 1.0) or 1.0),
                "snap_weight": float(getattr(defender, "snap_weight", 1.0) or 1.0),
            }

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        if self.players:
            pl.DataFrame(list(self.players.values())).sort(
                ["team", "position", "player_name"]
            ).write_csv(out / "player_skill_snapshot.csv")
        if self.defenders:
            pl.DataFrame(list(self.defenders.values())).sort(
                ["position", "player_id"]
            ).write_csv(out / "defender_skill_snapshot.csv")
