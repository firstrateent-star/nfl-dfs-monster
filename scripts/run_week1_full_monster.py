from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.ol_simulation import compile_ol_simulation_context
from monster.feature_compile.skill_pools import compile_current_skill_pools, compile_player_physical_inputs
from monster.feature_compile.units import apply_team_unit_effects, compile_team_unit_effects
from monster.sim.pipeline import simulate_monster_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState

MATCHUPS = (
    ("CHI", "CAR"), ("BUF", "HOU"), ("NO", "DET"), ("CLE", "JAX"),
    ("TB", "CIN"), ("ATL", "PIT"), ("NYJ", "TEN"), ("BAL", "IND"),
    ("ARI", "LAC"), ("WAS", "PHI"), ("MIA", "LV"), ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)
PARENT_TOTAL_REFERENCE = {
    "CHI@CAR": 48.7, "BUF@HOU": 49.0, "NO@DET": 49.2, "CLE@JAX": 40.265085,
    "TB@CIN": 50.8, "ATL@PIT": 42.009865, "NYJ@TEN": 40.467785,
    "BAL@IND": 49.6, "ARI@LAC": 48.1, "WAS@PHI": 45.41134,
    "MIA@LV": 39.939785, "GB@MIN": 41.2,
}


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(float), q))


def _unit_context(personnel: pl.DataFrame, historical_ol: pl.DataFrame | None):
    unit_map = compile_league_unit_player_map(personnel)
    rows = []
    for team_id, players in unit_map.items():
        effects, _ = compile_team_unit_effects(players)
        rows.append({"team_id": team_id, **effects.__dict__})
    effects_frame = pl.DataFrame(rows)
    ol = compile_ol_simulation_context(personnel, effects_frame, historical_ol)
    return unit_map, {str(row["team_id"]): row for row in ol.to_dicts()}


def _strengthened_state(base, unit_players, ol_row):
    effects, _ = compile_team_unit_effects(unit_players)
    state = apply_team_unit_effects(base, effects)
    return replace(
        state,
        offensive_line_continuity=float(ol_row["offensive_line_continuity"]),
        offensive_line_uncertainty=float(ol_row["offensive_line_uncertainty"]),
    )


def _player_distribution_rows(game_name: str, team_id: str, pool, side) -> list[dict]:
    rows = []
    for player in pool.players:
        stats = side.player_stats[player.player_id]
        fd_partial = (
            stats["passing_yards"] * 0.04
            + stats["passing_tds"] * 4.0
            + stats["rushing_yards"] * 0.10
            + stats["rushing_tds"] * 6.0
            + stats["receptions"] * 0.50
            + stats["receiving_yards"] * 0.10
            + stats["receiving_tds"] * 6.0
        )
        row = {
            "game": game_name, "team_id": team_id, "player_id": player.player_id,
            "player": player.display_name, "position": player.position,
            "active_probability": player.active_probability,
            "effectiveness_if_active": player.effectiveness_if_active,
            "role_uncertainty": player.role_uncertainty,
            "fd_points_before_turnover_penalties_mean": float(fd_partial.mean()),
            "fd_points_before_turnover_penalties_p20": _q(fd_partial, 0.20),
            "fd_points_before_turnover_penalties_p50": _q(fd_partial, 0.50),
            "fd_points_before_turnover_penalties_p90": _q(fd_partial, 0.90),
            "fd_points_before_turnover_penalties_p95": _q(fd_partial, 0.95),
            "fd_points_before_turnover_penalties_p99": _q(fd_partial, 0.99),
        }
        for stat in (
            "pass_attempts", "passing_yards", "passing_tds", "targets", "receptions",
            "receiving_yards", "receiving_tds", "rush_attempts", "rushing_yards", "rushing_tds",
        ):
            values = stats[stat]
            row[f"{stat}_mean"] = float(values.mean())
            row[f"{stat}_p90"] = _q(values, 0.90)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--worlds", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=2026090710)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-full-monster"))
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

    game_rows = []
    player_rows = []
    qb_rows = []
    for idx, (away, home) in enumerate(MATCHUPS):
        game_name = f"{away}@{home}"
        states = compile_team_state_map(policy, {away: home, home: away})
        away_state = _strengthened_state(states[away], unit_map[away], ol_map[away])
        home_state = _strengthened_state(states[home], unit_map[home], ol_map[home])
        seed = args.seed + idx * 10_007
        result = simulate_monster_game(
            GameState(game_name, away_state, home_state), pools[away], pools[home],
            worlds=args.worlds, seed=seed, player_inputs=physical_inputs,
        )
        g = result.game_worlds
        total = g.away_points + g.home_points
        margin = g.away_points - g.home_points
        model_total = float(total.mean())
        reference = PARENT_TOTAL_REFERENCE[game_name]
        game_rows.append({
            "game": game_name, "seed": seed, "worlds": args.worlds,
            "away_points_mean": float(g.away_points.mean()),
            "home_points_mean": float(g.home_points.mean()),
            "total_mean": model_total, "parent_v0_6_2_total_reference": reference,
            "delta_vs_parent_reference": model_total - reference,
            "total_p10": _q(total, 0.10), "total_p20": _q(total, 0.20),
            "total_p50": _q(total, 0.50), "total_p80": _q(total, 0.80),
            "total_p90": _q(total, 0.90), "p60_plus": float((total >= 60).mean()),
            "margin_mean": float(margin.mean()), "close_within_7": float((np.abs(margin) <= 7).mean()),
            "away_td_mean": float(g.away_touchdowns.mean()), "home_td_mean": float(g.home_touchdowns.mean()),
            "away_fg_mean": float(g.away_field_goals.mean()), "home_fg_mean": float(g.home_field_goals.mean()),
            "away_turnovers_mean": float(g.away_turnovers.mean()), "home_turnovers_mean": float(g.home_turnovers.mean()),
        })
        player_rows.extend(_player_distribution_rows(game_name, away, result.snapshot.away_pool, result.allocation_worlds.away))
        player_rows.extend(_player_distribution_rows(game_name, home, result.snapshot.home_pool, result.allocation_worlds.home))
        for team_id, pool in ((away, result.snapshot.away_pool), (home, result.snapshot.home_pool)):
            for player in pool.players:
                if player.position == "QB":
                    qb_rows.append({
                        "game": game_name, "team_id": team_id, "player": player.display_name,
                        "qb_pass_share": player.qb_pass_share, "active_probability": player.active_probability,
                        "effectiveness_if_active": player.effectiveness_if_active,
                    })

    games = pl.DataFrame(game_rows).sort("total_mean", descending=True)
    players = pl.DataFrame(player_rows).sort("fd_points_before_turnover_penalties_mean", descending=True)
    qbs = pl.DataFrame(qb_rows).sort(["team_id", "qb_pass_share"], descending=[False, True])
    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "game_distribution_map.csv")
    players.write_csv(args.out / "player_distributions.csv")
    qbs.write_csv(args.out / "qb_state_audit.csv")
    personnel.filter(
        (pl.col("health_availability_probability") < 0.95)
        | (pl.col("health_effectiveness_if_active") < 0.98)
    ).write_csv(args.out / "health_states_used.csv")

    manifest = {
        "artifact": "Monster Week 1 Full Football Simulation — Health + OL Strengthened",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game": args.worlds,
        "base_seed": args.seed,
        "games": [f"{a}@{h}" for a, h in MATCHUPS],
        "market_blind_production": True,
        "parent_total_reference_is_audit_only": True,
        "salary_used_in_football_simulation": False,
        "ownership_used_in_football_simulation": False,
        "health_dimensions": ["availability", "effectiveness_if_active", "health_uncertainty"],
        "ol_state_enabled": True,
        "physical_mechanism_inputs_enabled": True,
        "fantasy_points_note": "FD subtotal excludes interception and fumble penalties because player-level turnover attribution is not yet modeled.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(games)
    print(players.head(40))
    print(qbs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
