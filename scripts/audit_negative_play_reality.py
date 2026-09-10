from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl
from run_week1_v13_first_sim import (
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
from monster.ingest.nflverse import configure_cache
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map

GAME_DATE = datetime(2026, 9, 13, tzinfo=UTC).date()
FAMILIES = (
    "all_scrimmage",
    "all_dropbacks",
    "all_rush_attempts",
    "designed_run",
    "scramble",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "sack",
)


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(row: dict[str, Any], name: str) -> bool:
    return _number(row, name) == 1.0


def _historical_family(row: dict[str, Any]) -> str | None:
    dropback = _flag(row, "qb_dropback")
    if not dropback:
        return "designed_run" if _flag(row, "rush_attempt") else None
    if _flag(row, "sack"):
        return "sack"
    if _flag(row, "qb_scramble"):
        return "scramble"
    if _flag(row, "interception"):
        return "interception"
    if _flag(row, "complete_pass"):
        return "complete_pass"
    return "incomplete_pass"


def _sim_family(event: PlayEvent) -> str | None:
    if event.play_type == PlayType.RUN:
        return "designed_run"
    if event.play_type != PlayType.PASS:
        return None
    return {
        PassResult.SCRAMBLE: "scramble",
        PassResult.COMPLETE: "complete_pass",
        PassResult.INCOMPLETE: "incomplete_pass",
        PassResult.INTERCEPTION: "interception",
        PassResult.SACK: "sack",
    }.get(event.pass_result)


def _append(store: dict[str, list[float]], family: str, yards: float) -> None:
    store.setdefault(family, []).append(float(yards))
    store.setdefault("all_scrimmage", []).append(float(yards))
    if family in {"designed_run", "scramble"}:
        store.setdefault("all_rush_attempts", []).append(float(yards))
    if family in {"scramble", "complete_pass", "incomplete_pass", "interception", "sack"}:
        store.setdefault("all_dropbacks", []).append(float(yards))


def _quantile(arr: np.ndarray, q: float) -> float | None:
    return float(np.quantile(arr, q)) if arr.size else None


def _summary(store: dict[str, list[float]], *, source: str, scope: str) -> list[dict[str, object]]:
    total = len(store.get("all_scrimmage", []))
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        values = store.get(family, [])
        arr = np.asarray(values, dtype=float)
        events = int(arr.size)
        negative = arr[arr < 0.0]
        rows.append(
            {
                "source": source,
                "scope": scope,
                "family": family,
                "events": events,
                "share_of_scrimmage": events / total if total else 0.0,
                "yards_mean": float(arr.mean()) if events else None,
                "yards_p10": _quantile(arr, 0.10),
                "yards_p25": _quantile(arr, 0.25),
                "yards_p50": _quantile(arr, 0.50),
                "negative_gain_rate": float(np.mean(arr < 0.0)) if events else None,
                "zero_gain_rate": float(np.mean(arr == 0.0)) if events else None,
                "nonpositive_gain_rate": float(np.mean(arr <= 0.0)) if events else None,
                "loss_2plus_rate": float(np.mean(arr <= -2.0)) if events else None,
                "loss_5plus_rate": float(np.mean(arr <= -5.0)) if events else None,
                "gain_1_to_3_rate": float(np.mean((arr > 0.0) & (arr <= 3.0))) if events else None,
                "gain_4_to_14_rate": float(np.mean((arr >= 4.0) & (arr < 15.0))) if events else None,
                "gain_15plus_rate": float(np.mean(arr >= 15.0)) if events else None,
                "conditional_loss_mean": float(negative.mean()) if negative.size else None,
            }
        )
    return rows


def _comparison(simulated: list[dict[str, object]], historical: list[dict[str, object]]) -> list[dict[str, object]]:
    hist = {str(row["family"]): row for row in historical}
    metrics = (
        "share_of_scrimmage",
        "yards_mean",
        "yards_p10",
        "yards_p25",
        "negative_gain_rate",
        "zero_gain_rate",
        "nonpositive_gain_rate",
        "loss_2plus_rate",
        "loss_5plus_rate",
        "gain_1_to_3_rate",
        "gain_4_to_14_rate",
        "gain_15plus_rate",
        "conditional_loss_mean",
    )
    rows: list[dict[str, object]] = []
    for sim in simulated:
        family = str(sim["family"])
        hist_row = hist.get(family)
        if hist_row is None:
            continue
        for metric in metrics:
            sim_value = sim.get(metric)
            hist_value = hist_row.get(metric)
            if sim_value is None or hist_value is None:
                continue
            sim_f = float(sim_value)
            hist_f = float(hist_value)
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "simulated": sim_f,
                    "historical": hist_f,
                    "absolute_delta": sim_f - hist_f,
                    "relative_delta": (sim_f - hist_f) / hist_f if abs(hist_f) > 1e-12 else None,
                }
            )
    return rows


def _lookup(comparison: list[dict[str, object]], family: str, metric: str) -> dict[str, object]:
    for row in comparison:
        if row["family"] == family and row["metric"] == metric:
            return row
    raise KeyError((family, metric))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026091022)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/negative-play-reality"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
    )
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

    simulated_store: dict[str, list[float]] = {}
    simulated_by_game: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        game_store: dict[str, list[float]] = {}
        simulated_by_game[game] = game_store
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            for event in result.plays:
                family = _sim_family(event)
                if family is None:
                    continue
                _append(simulated_store, family, event.yards)
                _append(game_store, family, event.yards)

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)
    if "no_play" in pbp.columns:
        pbp = pbp.filter(pl.col("no_play").fill_null(0) == 0)

    historical_store: dict[str, list[float]] = {}
    for row in pbp.to_dicts():
        family = _historical_family(row)
        if family is None:
            continue
        _append(historical_store, family, _number(row, "yards_gained"))

    simulated = _summary(simulated_store, source="monster_v1_3", scope="2026_week1_12_game_slate")
    historical = _summary(historical_store, source="nflverse", scope=f"{args.history}_regular_season")
    by_game = [
        row
        for game, store in simulated_by_game.items()
        for row in _summary(store, source="monster_v1_3", scope=game)
    ]
    comparison = _comparison(simulated, historical)

    diagnosis = {
        "all_scrimmage_negative_gain_relative_delta": _lookup(comparison, "all_scrimmage", "negative_gain_rate")["relative_delta"],
        "all_scrimmage_nonpositive_gain_relative_delta": _lookup(comparison, "all_scrimmage", "nonpositive_gain_rate")["relative_delta"],
        "designed_run_negative_gain_relative_delta": _lookup(comparison, "designed_run", "negative_gain_rate")["relative_delta"],
        "designed_run_loss_2plus_relative_delta": _lookup(comparison, "designed_run", "loss_2plus_rate")["relative_delta"],
        "scramble_negative_gain_relative_delta": _lookup(comparison, "scramble", "negative_gain_rate")["relative_delta"],
        "complete_pass_negative_gain_relative_delta": _lookup(comparison, "complete_pass", "negative_gain_rate")["relative_delta"],
        "complete_pass_zero_gain_relative_delta": _lookup(comparison, "complete_pass", "zero_gain_rate")["relative_delta"],
        "sack_conditional_loss_mean_delta": _lookup(comparison, "sack", "conditional_loss_mean")["absolute_delta"],
        "principle": "Negative and zero-yard outcomes are first-class football anatomy. This audit localizes missing failure mass; it does not authorize generic yard dampening.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(simulated).write_csv(args.out / "simulated_negative_play_summary.csv")
    pl.DataFrame(historical).write_csv(args.out / "historical_negative_play_summary.csv")
    pl.DataFrame(by_game).write_csv(args.out / "simulated_negative_play_by_game.csv")
    pl.DataFrame(comparison).write_csv(args.out / "negative_play_comparison.csv")
    manifest = {
        "artifact": "Monster v1.3 Negative/No-Gain Play Reality Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "simulation_season": 2026,
        "simulation_week": 1,
        "historical_season": args.history,
        "games": len(MATCHUPS),
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "goal_line_yardage_conservation_active": True,
        "historical_no_play_excluded": "no_play" in pbp.columns,
        "definitions": {
            "negative_gain": "credited scrimmage yards < 0",
            "zero_gain": "credited scrimmage yards == 0",
            "loss_2plus": "credited scrimmage yards <= -2",
            "loss_5plus": "credited scrimmage yards <= -5",
            "designed_run": "Monster RUN; nflverse rush_attempt == 1 and qb_dropback != 1",
            "complete_pass": "Monster PASS->COMPLETE; nflverse complete_pass == 1 after sack/scramble/interception exclusions",
            "historical_scope": "regular season, kneels/spikes excluded, provider no_play rows excluded when available",
        },
        "diagnosis": diagnosis,
        "files": {
            "simulated": "simulated_negative_play_summary.csv",
            "historical": "historical_negative_play_summary.csv",
            "by_game": "simulated_negative_play_by_game.csv",
            "comparison": "negative_play_comparison.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
