from __future__ import annotations
from pathlib import Path
import numpy as np
from monster.sim.game import simulate_game


def simulate_slate(games, worlds: int, base_seed: int, out_dir: Path) -> list[dict]:
    """Simulate each game independently so one changed game does not invalidate the other eleven."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, game in enumerate(games):
        result = simulate_game(game, worlds=worlds, seed=base_seed + i * 10_007)
        path = out_dir / f"{game.game_id}.npz"
        np.savez_compressed(path, away_points=result.away_points, home_points=result.home_points)
        total = result.total
        summaries.append({
            "game_id": game.game_id,
            "away_mean": float(result.away_points.mean()),
            "home_mean": float(result.home_points.mean()),
            "total_mean": float(total.mean()),
            "p10": float(np.quantile(total, .10)),
            "p50": float(np.quantile(total, .50)),
            "p90": float(np.quantile(total, .90)),
            "p60_plus": float((total >= 60).mean()),
            "artifact": str(path),
        })
    return summaries
