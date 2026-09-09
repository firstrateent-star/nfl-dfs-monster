from __future__ import annotations

from dataclasses import replace

import polars as pl

from monster.snapshot.player import TeamPlayerPool


def apply_health_to_skill_pools(
    pools: dict[str, TeamPlayerPool], personnel: pl.DataFrame
) -> dict[str, TeamPlayerPool]:
    """Apply health availability/effectiveness after role compilation.

    This intentionally overwrites any generic pool-level availability default with the
    explicit health-aware participation probability from the personnel snapshot. Role
    shares and role uncertainty remain untouched.
    """
    if "gsis_id" not in personnel.columns:
        return pools
    columns = ["gsis_id", "game_day_active_probability"]
    if "health_effectiveness_if_active" in personnel.columns:
        columns.append("health_effectiveness_if_active")
    rows = {
        str(row["gsis_id"]): row
        for row in personnel.select(columns).drop_nulls(["gsis_id"]).to_dicts()
    }

    out: dict[str, TeamPlayerPool] = {}
    for team_id, pool in pools.items():
        players = []
        for player in pool.players:
            health = rows.get(player.player_id)
            if health is None:
                players.append(player)
                continue
            active_probability = float(health.get("game_day_active_probability") or 0.0)
            effectiveness = float(health.get("health_effectiveness_if_active") or 1.0)
            players.append(
                replace(
                    player,
                    active_probability=max(0.0, min(1.0, active_probability)),
                    effectiveness_if_active=max(0.50, min(1.0, effectiveness)),
                )
            )
        out[team_id] = replace(pool, players=tuple(players))
    return out
