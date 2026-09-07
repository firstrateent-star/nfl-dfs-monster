from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
from monster.sim.game import simulate_game


def _artifact_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate_slate(games, worlds: int, base_seed: int, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, game in enumerate(games):
        seed = base_seed + i * 10_007
        result = simulate_game(game, worlds=worlds, seed=seed)
        path = out_dir / f"{game.game_id}.npz"
        np.savez_compressed(
            path,
            away_points=result.away_points, home_points=result.home_points,
            away_drives=result.away_drives, home_drives=result.home_drives,
            away_touchdowns=result.away_touchdowns, home_touchdowns=result.home_touchdowns,
            away_field_goals=result.away_field_goals, home_field_goals=result.home_field_goals,
            away_turnovers=result.away_turnovers, home_turnovers=result.home_turnovers,
        )
        total = result.total
        margin = result.away_points - result.home_points
        summaries.append({
            "game_id": game.game_id, "seed": seed, "worlds": worlds,
            "away_mean": float(result.away_points.mean()), "home_mean": float(result.home_points.mean()),
            "total_mean": float(total.mean()), "margin_mean": float(margin.mean()),
            "p10": float(np.quantile(total, 0.10)), "p50": float(np.quantile(total, 0.50)),
            "p90": float(np.quantile(total, 0.90)), "p60_plus": float((total >= 60).mean()),
            "close_7": float((np.abs(margin) <= 7).mean()),
            "away_drives_mean": float(result.away_drives.mean()), "home_drives_mean": float(result.home_drives.mean()),
            "away_td_mean": float(result.away_touchdowns.mean()), "home_td_mean": float(result.home_touchdowns.mean()),
            "artifact": str(path), "artifact_sha256": _artifact_hash(path),
        })
    return summaries
