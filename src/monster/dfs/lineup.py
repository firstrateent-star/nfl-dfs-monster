from __future__ import annotations

from dataclasses import dataclass

import polars as pl

SALARY_CAP = 60_000
OFFENSE = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX_POSITIONS = {"RB", "WR", "TE"}
DEFENSE_POSITIONS = {"D", "DEF", "DST"}


@dataclass(frozen=True)
class LineupAudit:
    legal: bool
    salary: int
    players: int
    reason: str | None = None


def audit_fanduel_lineup(lineup: pl.DataFrame, salary_cap: int = SALARY_CAP) -> LineupAudit:
    """Validate a FanDuel NFL classic lineup without using projections or model state."""
    required = {"position", "fanduel_id", "salary"}
    missing = required - set(lineup.columns)
    if missing:
        return LineupAudit(False, 0, lineup.height, f"missing columns: {sorted(missing)}")
    salary = int(lineup["salary"].sum()) if lineup.height else 0
    if lineup.height != 9:
        return LineupAudit(False, salary, lineup.height, "lineup must contain exactly 9 players")
    if lineup["fanduel_id"].n_unique() != 9:
        return LineupAudit(False, salary, lineup.height, "duplicate player")
    if salary > salary_cap:
        return LineupAudit(False, salary, lineup.height, "salary cap exceeded")

    positions = lineup["position"].cast(pl.String).str.to_uppercase().to_list()
    defense = sum(position in DEFENSE_POSITIONS for position in positions)
    if defense != 1:
        return LineupAudit(False, salary, lineup.height, "lineup must contain exactly one defense")
    offense_positions = [position for position in positions if position not in DEFENSE_POSITIONS]
    if offense_positions.count("QB") != 1:
        return LineupAudit(False, salary, lineup.height, "lineup must contain exactly one QB")
    for position, minimum in OFFENSE.items():
        if offense_positions.count(position) < minimum:
            return LineupAudit(False, salary, lineup.height, f"not enough {position}")
    extras = {position: offense_positions.count(position) - OFFENSE.get(position, 0) for position in FLEX_POSITIONS}
    if sum(extras.values()) != 1 or any(value < 0 for value in extras.values()):
        return LineupAudit(False, salary, lineup.height, "invalid RB/WR/TE flex construction")
    if any(position not in {"QB", "RB", "WR", "TE"} for position in offense_positions):
        return LineupAudit(False, salary, lineup.height, "invalid offensive position")
    return LineupAudit(True, salary, lineup.height)
