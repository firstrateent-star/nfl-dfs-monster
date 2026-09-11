from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from run_week1_v13_first_sim import (
    GAME_DATE,
    MATCHUPS,
    _defensive_unit,
    _read,
    _situational_context,
    _team_identity,
    _with_event_rush_plan,
)

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.sim.defensive_attribution import attribute_defensive_box_score
from monster.sim.event_ledger import assert_event_conservation
from monster.sim.game_loop_v13 import DefensiveBoxScore, PlayerBoxScore, simulate_game
from monster.sim.play_kernel import PlayEvent, PlayType
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _name_map(personnel: pl.DataFrame, pools) -> dict[str, str]:
    names = {p.player_id: p.display_name for pool in pools.values() for p in pool.players}
    id_col = next((c for c in ("gsis_id", "player_id", "pfr_id") if c in personnel.columns), None)
    name_col = next(
        (c for c in ("display_name", "full_name", "player_name", "football_name", "name") if c in personnel.columns),
        None,
    )
    if id_col is not None and name_col is not None:
        for row in personnel.select([id_col, name_col]).drop_nulls().to_dicts():
            names.setdefault(str(row[id_col]), str(row[name_col]))
    return names


def _position_map(personnel: pl.DataFrame, pools) -> dict[str, str]:
    positions = {p.player_id: p.position for pool in pools.values() for p in pool.players}
    id_col = next((c for c in ("gsis_id", "player_id", "pfr_id") if c in personnel.columns), None)
    position_col = next((c for c in ("position", "depth_position") if c in personnel.columns), None)
    if id_col is not None and position_col is not None:
        for row in personnel.select([id_col, position_col]).drop_nulls().to_dicts():
            positions.setdefault(str(row[id_col]), str(row[position_col]))
    return positions


def _team_map(pools, units) -> dict[str, str]:
    out: dict[str, str] = {}
    for team, pool in pools.items():
        for player in pool.players:
            out[player.player_id] = team
    for team, players in units.items():
        for player in players:
            out[player.player_id] = team
    return out


def _event_offense_team(event: PlayEvent, player_teams: dict[str, str]) -> str | None:
    for player_id in (event.passer_id, event.rusher_id, event.target_id, event.fumbler_id):
        if player_id is not None and player_id in player_teams:
            return player_teams[player_id]
    return None


def _defensive_plays(plays: tuple[PlayEvent, ...], opponent: str, player_teams: dict[str, str]) -> tuple[PlayEvent, ...]:
    return tuple(
        event
        for event in plays
        if event.play_type in {PlayType.RUN, PlayType.PASS}
        and _event_offense_team(event, player_teams) == opponent
    )


def _aggregate_player_rows(acc, names, positions, teams, *, side: str) -> list[dict]:
    rows: list[dict] = []
    for (game, player_id), metrics in acc.items():
        row = {
            "game": game,
            "team": teams.get(player_id, ""),
            "player_id": player_id,
            "player": names.get(player_id, player_id),
            "position": positions.get(player_id, ""),
            "side": side,
        }
        for stat, values in metrics.items():
            for label, value in _quantiles(values).items():
                row[f"{stat}_{label}"] = value
            arr = np.asarray(values, dtype=float)
            row[f"{stat}_1_plus_probability"] = float(np.mean(arr >= 1.0))
            if stat in {"passing_yards", "receiving_yards", "rushing_yards"}:
                row[f"{stat}_100_plus_probability"] = float(np.mean(arr >= 100.0))
        rows.append(row)
    return rows


def _representative_world(game_worlds: list[dict]) -> int:
    away_points = np.asarray([x["away_points"] for x in game_worlds], dtype=float)
    home_points = np.asarray([x["home_points"] for x in game_worlds], dtype=float)
    total_yards = np.asarray([x["total_yards"] for x in game_worlds], dtype=float)
    centers = np.asarray([np.median(away_points), np.median(home_points), np.median(total_yards)])
    scales = np.asarray([max(away_points.std(), 1.0), max(home_points.std(), 1.0), max(total_yards.std(), 1.0)])
    distances = []
    for x in game_worlds:
        vector = np.asarray([x["away_points"], x["home_points"], x["total_yards"]], dtype=float)
        distances.append(float(np.square((vector - centers) / scales).sum()))
    return int(game_worlds[int(np.argmin(distances))]["world"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026110911)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-v13-box-scores"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)
    pools = apply_health_to_skill_pools(compile_current_skill_pools(personnel, usage, policy=policy), personnel)
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))
    teams = {
        team: _team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in MATCHUPS
        for team in pair
    }
    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    names = _name_map(personnel, pools)
    positions = _position_map(personnel, pools)
    player_teams = _team_map(pools, units)

    offensive_acc = defaultdict(lambda: defaultdict(list))
    defensive_acc = defaultdict(lambda: defaultdict(list))
    game_acc = defaultdict(lambda: defaultdict(list))
    game_world_rows: dict[str, list[dict]] = defaultdict(list)
    representative_offense: dict[tuple[str, int, str], dict] = {}
    representative_defense: dict[tuple[str, int, str], dict] = {}

    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        offense_ids = {team: tuple(p.player_id for p in pools[team].players) for team in (away, home)}
        defense_ids = {
            team: tuple(p.player_id for p in units[team] if p.defense_snap_share >= 0.03)
            for team in (away, home)
        }
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(pools[away], rng=np.random.default_rng(seed + 101_003))
            home_plan = sample_event_rush_share_plan(pools[home], rng=np.random.default_rng(seed + 202_007))
            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            assert_event_conservation(result)

            team_box = {}
            for team, opponent, attribution_offset in (
                (away, home, 31_337),
                (home, away, 47_771),
            ):
                boxes = {pid: result.player_stats.get(pid, PlayerBoxScore()) for pid in offense_ids[team]}
                pass_yards = sum(x.passing_yards for x in boxes.values())
                rush_yards = sum(x.rushing_yards for x in boxes.values())
                team_box[team] = {
                    "pass_yards": float(pass_yards),
                    "rush_yards": float(rush_yards),
                    "total_yards": float(pass_yards + rush_yards),
                    "turnovers": float(sum(x.interceptions + x.fumbles_lost for x in boxes.values())),
                }
                for player_id, box in boxes.items():
                    data = asdict(box)
                    for stat, value in data.items():
                        offensive_acc[(game, player_id)][stat].append(float(value))
                    representative_offense[(game, world, player_id)] = data

                attributed = attribute_defensive_box_score(
                    _defensive_plays(result.plays, opponent, player_teams),
                    defenses[team],
                    seed=seed + attribution_offset,
                )
                for player_id in defense_ids[team]:
                    box = attributed.get(player_id, DefensiveBoxScore())
                    data = asdict(box)
                    for stat, value in data.items():
                        defensive_acc[(game, player_id)][stat].append(float(value))
                    representative_defense[(game, world, player_id)] = data

            away_points = float(result.final_state.away_score)
            home_points = float(result.final_state.home_score)
            for team, points, opponent_points in ((away, away_points, home_points), (home, home_points, away_points)):
                for stat, value in {"points": points, "points_allowed": opponent_points, **team_box[team]}.items():
                    game_acc[(game, team)][stat].append(float(value))
            game_world_rows[game].append(
                {
                    "world": world,
                    "away_points": away_points,
                    "home_points": home_points,
                    "total_yards": team_box[away]["total_yards"] + team_box[home]["total_yards"],
                }
            )

    offense_rows = _aggregate_player_rows(offensive_acc, names, positions, player_teams, side="offense")
    defense_rows = _aggregate_player_rows(defensive_acc, names, positions, player_teams, side="defense")
    team_rows: list[dict] = []
    for (game, team), metrics in game_acc.items():
        row = {"game": game, "team": team}
        for stat, values in metrics.items():
            for label, value in _quantiles(values).items():
                row[f"{stat}_{label}"] = value
        team_rows.append(row)

    representative_rows: list[dict] = []
    for game, worlds in game_world_rows.items():
        representative_world = _representative_world(worlds)
        away, home = game.split("@")
        world_summary = next(x for x in worlds if x["world"] == representative_world)
        representative_rows.append(
            {
                "game": game,
                "world": representative_world,
                "record_type": "game",
                "team": "",
                "player_id": "",
                "player": "",
                "position": "",
                "away_points": world_summary["away_points"],
                "home_points": world_summary["home_points"],
            }
        )
        for team in (away, home):
            for player_id in [p.player_id for p in pools[team].players]:
                representative_rows.append(
                    {
                        "game": game,
                        "world": representative_world,
                        "record_type": "offense",
                        "team": team,
                        "player_id": player_id,
                        "player": names.get(player_id, player_id),
                        "position": positions.get(player_id, ""),
                        **representative_offense[(game, representative_world, player_id)],
                    }
                )
            for player_id in [p.player_id for p in units[team] if p.defense_snap_share >= 0.03]:
                representative_rows.append(
                    {
                        "game": game,
                        "world": representative_world,
                        "record_type": "defense",
                        "team": team,
                        "player_id": player_id,
                        "player": names.get(player_id, player_id),
                        "position": positions.get(player_id, ""),
                        **representative_defense[(game, representative_world, player_id)],
                    }
                )

    args.out.mkdir(parents=True, exist_ok=True)
    offense_df = pl.DataFrame(offense_rows).sort(["game", "team", "position", "player"])
    defense_df = pl.DataFrame(defense_rows).sort(["game", "team", "tackles_mean"], descending=[False, False, True])
    team_df = pl.DataFrame(team_rows).sort(["game", "team"])
    representative_df = pl.DataFrame(representative_rows).sort(["game", "record_type", "team", "player"])
    offense_df.write_csv(args.out / "offensive_player_box_score_distributions.csv")
    defense_df.write_csv(args.out / "defensive_player_box_score_distributions.csv")
    team_df.write_csv(args.out / "team_box_score_distributions.csv")
    representative_df.write_csv(args.out / "representative_world_box_scores.csv")

    game_summaries = []
    for game in sorted(game_world_rows):
        away, home = game.split("@")
        game_summaries.append(
            {
                "game": game,
                "representative_world": _representative_world(game_world_rows[game]),
                "away": away,
                "home": home,
                "teams": [row for row in team_rows if row["game"] == game],
                "offense": [row for row in offense_rows if row["game"] == game],
                "defense": [row for row in defense_rows if row["game"] == game],
            }
        )
    (args.out / "simulated_game_box_scores.json").write_text(json.dumps(game_summaries, indent=2))
    manifest = {
        "model": "Monster v1.3 simulated game box-score observability layer",
        "season": 2026,
        "week": 1,
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "games": len(MATCHUPS),
        "market_blind_football": True,
        "dfs_inputs_used": False,
        "sportsbook_inputs_used": False,
        "offense_event_derived": True,
        "defense_event_derived": True,
        "role_aware_defensive_attribution": True,
        "zero_inclusive_worlds": True,
        "distribution_view": True,
        "representative_world_view": True,
        "files": {
            "team": "team_box_score_distributions.csv",
            "offense": "offensive_player_box_score_distributions.csv",
            "defense": "defensive_player_box_score_distributions.csv",
            "representative_world": "representative_world_box_scores.csv",
            "json": "simulated_game_box_scores.json",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
