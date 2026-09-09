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
    ("CHI", "CAR"), ("BUF", "HOU"), ("NO", "DET"), ("CLE", "JAC"),
    ("TB", "CIN"), ("ATL", "PIT"), ("NYJ", "TEN"), ("BAL", "IND"),
    ("ARI", "LAC"), ("WAS", "PHI"), ("MIA", "LV"), ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)


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
            "target_participation_probability": float((stats["targets"] > 0).mean()),
            "rush_participation_probability": float((stats["rush_attempts"] > 0).mean()),
            "pass_attempt_participation_probability": float((stats["pass_attempts"] > 0).mean()),
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
            row[f"{stat}_p50"] = _q(values, 0.50)
            row[f"{stat}_p90"] = _q(values, 0.90)
        rows.append(row)
    return rows


def _rank_world_metrics(matrix: np.ndarray, team_total: np.ndarray, prefix: str) -> dict[str, float]:
    """Rank opportunity recipients inside each world before averaging."""
    ranked = np.sort(matrix, axis=1)[:, ::-1] if matrix.shape[1] else matrix
    total = team_total.astype(float)
    out: dict[str, float] = {}
    for rank in range(3):
        label = rank + 1
        attempts = ranked[:, rank] if ranked.shape[1] > rank else np.zeros(matrix.shape[0], dtype=float)
        share = np.divide(attempts, np.maximum(total, 1.0))
        out[f"{prefix}_rank{label}_attempts_world_mean"] = float(attempts.mean())
        out[f"{prefix}_rank{label}_attempts_world_p50"] = _q(attempts, 0.50)
        out[f"{prefix}_rank{label}_attempts_world_p90"] = _q(attempts, 0.90)
        out[f"{prefix}_rank{label}_share_world_mean"] = float(share.mean())
        out[f"{prefix}_rank{label}_share_world_p50"] = _q(share, 0.50)
        out[f"{prefix}_rank{label}_share_world_p90"] = _q(share, 0.90)
    return out


def _team_opportunity_row(game_name: str, team_id: str, side, drives: np.ndarray) -> dict:
    plays = side.team_plays.astype(float)
    passing_yards = np.zeros(len(plays), dtype=float)
    rushing_yards = np.zeros(len(plays), dtype=float)
    rushing_columns = []
    target_columns = []
    for stats in side.player_stats.values():
        passing_yards += stats["passing_yards"]
        rushing_yards += stats["rushing_yards"]
        rushing_columns.append(stats["rush_attempts"].astype(float))
        target_columns.append(stats["targets"].astype(float))
    total_yards = passing_yards + rushing_yards
    yards_per_play = np.divide(total_yards, np.maximum(plays, 1.0))

    rushing_matrix = np.column_stack(rushing_columns) if rushing_columns else np.zeros((len(plays), 0), dtype=float)
    target_matrix = np.column_stack(target_columns) if target_columns else np.zeros((len(plays), 0), dtype=float)

    rushers = (rushing_matrix > 0).sum(axis=1)
    target_earners = (target_matrix > 0).sum(axis=1)
    one_target_earners = (target_matrix == 1).sum(axis=1)
    one_two_target_mask = (target_matrix > 0) & (target_matrix <= 2)
    one_two_target_earners = one_two_target_mask.sum(axis=1)
    peripheral_targets = np.where(one_two_target_mask, target_matrix, 0.0).sum(axis=1)
    team_targets = side.team_targets.astype(float)
    peripheral_target_share = np.divide(peripheral_targets, np.maximum(team_targets, 1.0))

    core_rushers = (rushing_matrix >= 3).sum(axis=1)
    incidental_mask = (rushing_matrix > 0) & (rushing_matrix <= 2)
    incidental_rushers = incidental_mask.sum(axis=1)
    incidental_attempts = np.where(incidental_mask, rushing_matrix, 0.0).sum(axis=1)
    team_rush_attempts = side.team_rush_attempts.astype(float)
    incidental_share = np.divide(incidental_attempts, np.maximum(team_rush_attempts, 1.0))

    row = {
        "game": game_name,
        "team_id": team_id,
        "drives_mean": float(drives.mean()),
        "plays_mean": float(plays.mean()),
        "plays_p10": _q(plays, 0.10),
        "plays_p50": _q(plays, 0.50),
        "plays_p90": _q(plays, 0.90),
        "dropbacks_mean": float(side.team_dropbacks.mean()),
        "sacks_mean": float(side.team_sacks.mean()),
        "pass_attempts_mean": float(side.team_pass_attempts.mean()),
        "targets_mean": float(side.team_targets.mean()),
        "target_earners_per_world_mean": float(target_earners.mean()),
        "target_earners_per_world_p10": _q(target_earners, 0.10),
        "target_earners_per_world_p50": _q(target_earners, 0.50),
        "target_earners_per_world_p90": _q(target_earners, 0.90),
        "one_target_earners_per_world_mean": float(one_target_earners.mean()),
        "one_two_target_earners_per_world_mean": float(one_two_target_earners.mean()),
        "peripheral_targets_per_world_mean": float(peripheral_targets.mean()),
        "peripheral_target_share_world_mean": float(peripheral_target_share.mean()),
        "rush_attempts_mean": float(side.team_rush_attempts.mean()),
        "rushers_per_world_mean": float(rushers.mean()),
        "rushers_per_world_p10": _q(rushers, 0.10),
        "rushers_per_world_p50": _q(rushers, 0.50),
        "rushers_per_world_p90": _q(rushers, 0.90),
        "core_rushers_per_world_mean": float(core_rushers.mean()),
        "core_rushers_per_world_p10": _q(core_rushers, 0.10),
        "core_rushers_per_world_p50": _q(core_rushers, 0.50),
        "core_rushers_per_world_p90": _q(core_rushers, 0.90),
        "incidental_rushers_per_world_mean": float(incidental_rushers.mean()),
        "incidental_rushers_per_world_p10": _q(incidental_rushers, 0.10),
        "incidental_rushers_per_world_p50": _q(incidental_rushers, 0.50),
        "incidental_rushers_per_world_p90": _q(incidental_rushers, 0.90),
        "incidental_rush_attempts_mean": float(incidental_attempts.mean()),
        "incidental_rush_share_mean": float(incidental_share.mean()),
        "passing_yards_mean": float(passing_yards.mean()),
        "rushing_yards_mean": float(rushing_yards.mean()),
        "total_yards_mean": float(total_yards.mean()),
        "total_yards_p10": _q(total_yards, 0.10),
        "total_yards_p50": _q(total_yards, 0.50),
        "total_yards_p90": _q(total_yards, 0.90),
        "yards_per_play_mean": float(yards_per_play.mean()),
        "yards_per_play_p90": _q(yards_per_play, 0.90),
    }
    row.update(_rank_world_metrics(rushing_matrix, side.team_rush_attempts, "rush"))
    row.update(_rank_world_metrics(target_matrix, side.team_targets, "target"))
    return row


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
    opportunity_rows = []
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
        game_rows.append({
            "game": game_name, "seed": seed, "worlds": args.worlds,
            "away_points_mean": float(g.away_points.mean()),
            "home_points_mean": float(g.home_points.mean()),
            "total_mean": float(total.mean()),
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
        opportunity_rows.append(_team_opportunity_row(game_name, away, result.allocation_worlds.away, g.away_drives))
        opportunity_rows.append(_team_opportunity_row(game_name, home, result.allocation_worlds.home, g.home_drives))
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
    opportunities = pl.DataFrame(opportunity_rows).sort("total_yards_mean", descending=True)
    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "game_distribution_map.csv")
    players.write_csv(args.out / "player_distributions.csv")
    qbs.write_csv(args.out / "qb_state_audit.csv")
    opportunities.write_csv(args.out / "team_opportunity_map.csv")
    personnel.filter(
        (pl.col("health_availability_probability") < 0.95)
        | (pl.col("health_effectiveness_if_active") < 0.98)
    ).write_csv(args.out / "health_states_used.csv")

    manifest = {
        "artifact": "Monster Week 1 Full Football Simulation — Continuity + Health + OL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game": args.worlds,
        "base_seed": args.seed,
        "games": [f"{a}@{h}" for a, h in MATCHUPS],
        "market_blind_production": True,
        "market_or_contaminated_reference_loaded": False,
        "salary_used_in_football_simulation": False,
        "ownership_used_in_football_simulation": False,
        "continuity_conditioned_policy": True,
        "health_dimensions": ["availability", "effectiveness_if_active", "health_uncertainty"],
        "ol_state_enabled": True,
        "physical_mechanism_inputs_enabled": True,
        "team_opportunity_audit_enabled": True,
        "player_participation_probabilities_enabled": True,
        "world_level_rushing_role_structure_enabled": True,
        "world_level_rushing_rank_audit_enabled": True,
        "world_level_target_rank_audit_enabled": True,
        "world_level_target_tail_audit_enabled": True,
        "fantasy_points_note": "FD subtotal excludes interception and fumble penalties because player-level turnover attribution is not yet modeled.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(games)
    print(opportunities)
    print(players.head(40))
    print(qbs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
