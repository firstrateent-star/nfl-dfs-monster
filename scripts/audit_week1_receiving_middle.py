from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.skill_pools import compile_current_skill_pools, compile_player_physical_inputs
from monster.sim.pipeline import simulate_monster_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from scripts.run_week1_full_monster import (
    GAME_DATE,
    MATCHUPS,
    _read,
    _strengthened_state,
    _unit_context,
)


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(float), q))


def _historical_target_tiers(season: int) -> dict[str, float]:
    import nflreadpy as nfl

    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    keep = ["game_id", "posteam", "pass_attempt", "receiver_player_id", "qb_spike"]
    pbp = pbp.select([c for c in keep if c in pbp.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    target_play = (
        (pl.col("pass_attempt").fill_null(0) == 1)
        & pl.col("receiver_player_id").is_not_null()
    )
    player_game = (
        pbp.filter(target_play)
        .group_by(["game_id", "posteam", "receiver_player_id"])
        .agg(pl.len().alias("targets"))
    )
    team_game = (
        player_game.group_by(["game_id", "posteam"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.len().alias("earners"),
            ((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_earners"),
            pl.col("targets").filter((pl.col("targets") >= 3) & (pl.col("targets") <= 5)).sum().alias("middle_targets"),
        )
        .with_columns(
            (pl.col("middle_targets") / pl.col("team_targets").clip(lower_bound=1)).alias("middle_share")
        )
    )
    return {
        "team_games": float(team_game.height),
        "target_earners": float(team_game["earners"].mean()),
        "middle_earners": float(team_game["middle_earners"].mean()),
        "middle_targets": float(team_game["middle_targets"].mean()),
        "middle_share": float(team_game["middle_share"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--worlds", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026090710)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-receiving-middle"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    historical_ol = _read(args.ol_outcomes) if args.ol_outcomes and args.ol_outcomes.exists() else None

    unit_map, ol_map = _unit_context(personnel, historical_ol)
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
    )
    physical_inputs = compile_player_physical_inputs(personnel, game_date=GAME_DATE)

    world_middle_earners: list[float] = []
    world_middle_targets: list[float] = []
    world_middle_shares: list[float] = []
    world_earners: list[float] = []
    team_rows: list[dict] = []

    for idx, (away, home) in enumerate(MATCHUPS):
        states = compile_team_state_map(policy, {away: home, home: away})
        away_state = _strengthened_state(states[away], unit_map[away], ol_map[away])
        home_state = _strengthened_state(states[home], unit_map[home], ol_map[home])
        result = simulate_monster_game(
            GameState(f"{away}@{home}", away_state, home_state),
            pools[away], pools[home],
            worlds=args.worlds,
            seed=args.seed + idx * 10_007,
            player_inputs=physical_inputs,
        )
        for team_id, side in ((away, result.allocation_worlds.away), (home, result.allocation_worlds.home)):
            matrix = np.column_stack(
                [stats["targets"].astype(float) for stats in side.player_stats.values()]
            )
            earners = (matrix > 0).sum(axis=1)
            middle_mask = (matrix >= 3) & (matrix <= 5)
            middle_earners = middle_mask.sum(axis=1)
            middle_targets = np.where(middle_mask, matrix, 0.0).sum(axis=1)
            middle_share = np.divide(
                middle_targets,
                np.maximum(side.team_targets.astype(float), 1.0),
            )
            world_earners.extend(earners.astype(float))
            world_middle_earners.extend(middle_earners.astype(float))
            world_middle_targets.extend(middle_targets.astype(float))
            world_middle_shares.extend(middle_share.astype(float))
            team_rows.append({
                "team_id": team_id,
                "target_earners": float(earners.mean()),
                "middle_earners": float(middle_earners.mean()),
                "middle_targets": float(middle_targets.mean()),
                "middle_share": float(middle_share.mean()),
            })

    historical = _historical_target_tiers(args.history)
    simulated = {
        "target_earners": float(np.mean(world_earners)),
        "middle_earners": float(np.mean(world_middle_earners)),
        "middle_targets": float(np.mean(world_middle_targets)),
        "middle_share": float(np.mean(world_middle_shares)),
    }
    rows = [
        {
            "metric": metric,
            "historical": historical[metric],
            "simulated": simulated[metric],
            "delta": simulated[metric] - historical[metric],
        }
        for metric in ("target_earners", "middle_earners", "middle_targets", "middle_share")
    ]
    comparison = pl.DataFrame(rows)

    # Candidate bands are deliberately broad enough to validate structure without fitting Week 1.
    middle_tier_within_candidate_band = (
        abs(simulated["middle_earners"] - historical["middle_earners"]) <= 0.45
        and abs(simulated["middle_share"] - historical["middle_share"]) <= 0.045
    )

    args.out.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(args.out / "receiving_middle_comparison.csv")
    pl.DataFrame(team_rows).write_csv(args.out / "team_receiving_middle.csv")
    manifest = {
        "artifact": "Monster Week 1 Receiving Middle-Tier Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "worlds_per_game": args.worlds,
        "simulated_team_games": len(team_rows),
        "historical": historical,
        "simulated": simulated,
        "middle_tier_within_candidate_band": bool(middle_tier_within_candidate_band),
        "market_blind": True,
        "principle": "A receiving hierarchy is not promoted merely for fixing rank 1; it must also reproduce the stable 3-5 target middle layer without consuming the already-valid 1-2 target tail.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
