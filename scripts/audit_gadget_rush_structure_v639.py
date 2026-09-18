from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from runtime_v639_composer import build_week1_runtime_inputs_v639

from monster.sim import reality_v62
from monster.sim.current_role_guard_v639 import (
    _entry_probability_v639,
    _rotation_exposure_v639,
)
from monster.sim.gadget_rush_entry_priors_v639 import (
    carry_bin,
    empirical_gadget_entry_prior,
)

HISTORICAL_REFERENCE = {
    "seasons": [2022, 2023, 2024, 2025],
    "mean_gadget_team_rush_share": 0.03674310517636209,
    "median_gadget_team_rush_share": 0.02702702702702703,
    "p90_gadget_team_rush_share": 0.09732494103454689,
    "probability_team_has_any_gadget_rush": 0.5266789328426863,
    "mean_gadget_rushers_per_team_week": 0.6743330266789328,
    "wr_player_game_entry_rate": 0.1315333672949567,
    "te_player_game_entry_rate": 0.03567787971457696,
}


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=6381701)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/v639-gadget-structure"),
    )
    args = parser.parse_args()

    runtime = build_week1_runtime_inputs_v639(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )
    pools = runtime["pools"]

    rows: list[dict[str, object]] = []
    player_summaries: list[dict[str, object]] = []
    team_summaries: list[dict[str, object]] = []
    all_gadget_mass: list[float] = []
    all_gadget_count: list[int] = []

    for team_index, team_id in enumerate(sorted(pools)):
        pool = pools[team_id]
        position_by_id = {
            player.player_id: player.position.upper()
            for player in pool.players
        }
        player_by_id = {player.player_id: player for player in pool.players}
        gadget_player_ids = [
            player.player_id
            for player in pool.players
            if player.position.upper() in {"WR", "TE"}
        ]
        player_entry_counts = {player_id: 0 for player_id in gadget_player_ids}
        player_share_sums = {player_id: 0.0 for player_id in gadget_player_ids}
        gadget_mass_worlds: list[float] = []
        gadget_count_worlds: list[int] = []
        wr_count_worlds: list[int] = []
        te_count_worlds: list[int] = []

        for world in range(args.worlds):
            rng = np.random.default_rng(
                args.seed + team_index * 1_000_003 + world * 101
            )
            plan = reality_v62.sample_event_rush_share_plan(pool, rng=rng)
            wr_ids = [
                player_id
                for player_id, share in plan.items()
                if share > 0.0 and position_by_id.get(player_id) == "WR"
            ]
            te_ids = [
                player_id
                for player_id, share in plan.items()
                if share > 0.0 and position_by_id.get(player_id) == "TE"
            ]
            gadget_ids = wr_ids + te_ids
            for player_id in gadget_ids:
                player_entry_counts[player_id] += 1
                player_share_sums[player_id] += float(plan[player_id])
            gadget_mass = sum(float(plan[player_id]) for player_id in gadget_ids)
            gadget_count = len(gadget_ids)

            gadget_mass_worlds.append(gadget_mass)
            gadget_count_worlds.append(gadget_count)
            wr_count_worlds.append(len(wr_ids))
            te_count_worlds.append(len(te_ids))
            all_gadget_mass.append(gadget_mass)
            all_gadget_count.append(gadget_count)

            rows.append(
                {
                    "team_id": team_id,
                    "world": world,
                    "gadget_rush_share": gadget_mass,
                    "gadget_rusher_count": gadget_count,
                    "wr_gadget_rusher_count": len(wr_ids),
                    "te_gadget_rusher_count": len(te_ids),
                }
            )

        for player_id in gadget_player_ids:
            player = player_by_id[player_id]
            player_summaries.append(
                {
                    "team_id": team_id,
                    "player_id": player_id,
                    "player": player.display_name,
                    "position": player.position.upper(),
                    "historical_rushes": float(player.historical_rushes),
                    "historical_rush_share": float(player.historical_rush_share),
                    "heuristic_rush_role_probability": float(
                        player.rush_role_probability
                    ),
                    "historical_carry_bin": carry_bin(player.historical_rushes),
                    "empirical_entry_prior": empirical_gadget_entry_prior(
                        player.position,
                        player.historical_rushes,
                    ),
                    "rotation_exposure": _rotation_exposure_v639(player_id),
                    "final_entry_probability": _entry_probability_v639(player),
                    "simulated_entry_rate": player_entry_counts[player_id] / args.worlds,
                    "mean_plan_share_all_worlds": player_share_sums[player_id] / args.worlds,
                }
            )

        team_summaries.append(
            {
                "team_id": team_id,
                "worlds": args.worlds,
                "mean_gadget_rush_share": float(np.mean(gadget_mass_worlds)),
                "median_gadget_rush_share": float(np.median(gadget_mass_worlds)),
                "p90_gadget_rush_share": _quantile(gadget_mass_worlds, 0.90),
                "probability_any_gadget_rush": float(
                    np.mean(np.asarray(gadget_count_worlds) > 0)
                ),
                "mean_gadget_rushers": float(np.mean(gadget_count_worlds)),
                "mean_wr_gadget_rushers": float(np.mean(wr_count_worlds)),
                "mean_te_gadget_rushers": float(np.mean(te_count_worlds)),
            }
        )

    league = {
        "team_worlds": len(all_gadget_mass),
        "mean_gadget_team_rush_share": float(np.mean(all_gadget_mass)),
        "median_gadget_team_rush_share": float(np.median(all_gadget_mass)),
        "p90_gadget_team_rush_share": _quantile(all_gadget_mass, 0.90),
        "probability_team_has_any_gadget_rush": float(
            np.mean(np.asarray(all_gadget_count) > 0)
        ),
        "mean_gadget_rushers_per_team_world": float(
            np.mean(all_gadget_count)
        ),
    }
    deltas = {
        key: float(league[key] - HISTORICAL_REFERENCE[key])
        for key in (
            "mean_gadget_team_rush_share",
            "median_gadget_team_rush_share",
            "p90_gadget_team_rush_share",
            "probability_team_has_any_gadget_rush",
        )
    }
    deltas["mean_gadget_rushers"] = float(
        league["mean_gadget_rushers_per_team_world"]
        - HISTORICAL_REFERENCE["mean_gadget_rushers_per_team_week"]
    )

    manifest = {
        "artifact": "Monster v6.3.9 Pregame Gadget Rush Structure Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "worlds_per_team": args.worlds,
        "seed": args.seed,
        "week1_2026_truth_used": False,
        "market_blind": True,
        "runtime_fingerprint": runtime["fingerprint"].__dict__,
        "historical_reference": HISTORICAL_REFERENCE,
        "league_simulated": league,
        "simulated_minus_historical": deltas,
        "principle": (
            "Validate role-world geometry against pre-forecast multi-season football "
            "structure before looking at realized Week 1 outcomes."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    import polars as pl

    pl.DataFrame(rows).write_csv(args.out / "team_worlds.csv")
    pl.DataFrame(player_summaries).write_csv(args.out / "player_summary.csv")
    pl.DataFrame(team_summaries).write_csv(args.out / "team_summary.csv")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
