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


def _bucket_store() -> defaultdict[str, list[float]]:
    return defaultdict(list)


def _append_bucket(
    store: dict[str, defaultdict[str, list[float]]],
    family: str,
    *,
    yards: float,
    touchdown: bool,
    air_yards: float | None = None,
    yac: float | None = None,
    yards_before_contact: float | None = None,
    yards_after_contact: float | None = None,
) -> None:
    bucket = store.setdefault(family, _bucket_store())
    bucket["yards"].append(float(yards))
    bucket["touchdown"].append(float(touchdown))
    if air_yards is not None:
        bucket["air_yards"].append(float(air_yards))
    if yac is not None:
        bucket["yac"].append(float(yac))
    if yards_before_contact is not None:
        bucket["yards_before_contact"].append(float(yards_before_contact))
    if yards_after_contact is not None:
        bucket["yards_after_contact"].append(float(yards_after_contact))


def _append_sim_event(
    store: dict[str, defaultdict[str, list[float]]],
    event: PlayEvent,
) -> None:
    if event.play_type == PlayType.RUN:
        family = "designed_run"
    elif event.play_type == PlayType.PASS:
        family = {
            PassResult.SCRAMBLE: "scramble",
            PassResult.COMPLETE: "complete_pass",
            PassResult.INCOMPLETE: "incomplete_pass",
            PassResult.INTERCEPTION: "interception",
            PassResult.SACK: "sack",
        }.get(event.pass_result)
        if family is None:
            raise ValueError(f"pass event missing recognized terminal result: {event.pass_result}")
    else:
        return

    kwargs = {
        "yards": float(event.yards),
        "touchdown": bool(event.touchdown),
        "air_yards": float(event.air_yards) if family == "complete_pass" else None,
        "yac": float(event.yards_after_catch) if family == "complete_pass" else None,
        "yards_before_contact": (
            float(event.yards_before_contact)
            if family in {"designed_run", "scramble"}
            else None
        ),
        "yards_after_contact": (
            float(event.yards_after_contact)
            if family in {"designed_run", "scramble"}
            else None
        ),
    }
    _append_bucket(store, family, **kwargs)
    _append_bucket(store, "all_scrimmage", **kwargs)
    if event.play_type == PlayType.PASS:
        _append_bucket(store, "all_dropbacks", **kwargs)
    if family in {"designed_run", "scramble"}:
        _append_bucket(store, "all_rush_attempts", **kwargs)


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


def _append_historical_row(
    store: dict[str, defaultdict[str, list[float]]],
    row: dict[str, Any],
) -> None:
    family = _historical_family(row)
    if family is None:
        return
    yards = _number(row, "yards_gained")
    touchdown = _flag(row, "pass_touchdown") or _flag(row, "rush_touchdown")
    air_yards = None
    yac = None
    if family == "complete_pass":
        if row.get("air_yards") is not None:
            air_yards = _number(row, "air_yards")
        if row.get("yards_after_catch") is not None:
            yac = _number(row, "yards_after_catch")
    kwargs = {
        "yards": yards,
        "touchdown": touchdown,
        "air_yards": air_yards,
        "yac": yac,
    }
    _append_bucket(store, family, **kwargs)
    _append_bucket(store, "all_scrimmage", **kwargs)
    if _flag(row, "qb_dropback"):
        _append_bucket(store, "all_dropbacks", **kwargs)
    if family in {"designed_run", "scramble"}:
        _append_bucket(store, "all_rush_attempts", **kwargs)


def _stat(values: list[float], kind: str) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if kind == "mean":
        return float(arr.mean())
    if kind == "p50":
        return float(np.quantile(arr, 0.50))
    if kind == "p90":
        return float(np.quantile(arr, 0.90))
    raise ValueError(kind)


def _summaries(
    store: dict[str, defaultdict[str, list[float]]],
    *,
    source: str,
    scope: str,
) -> list[dict[str, object]]:
    total = len(store.get("all_scrimmage", {}).get("yards", []))
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        bucket = store.get(family, _bucket_store())
        yards = bucket.get("yards", [])
        touchdowns = bucket.get("touchdown", [])
        arr = np.asarray(yards, dtype=float) if yards else np.asarray([], dtype=float)
        events = len(yards)
        row: dict[str, object] = {
            "source": source,
            "scope": scope,
            "family": family,
            "events": events,
            "share_of_scrimmage": events / total if total else 0.0,
            "yards_mean": _stat(yards, "mean"),
            "yards_p50": _stat(yards, "p50"),
            "yards_p90": _stat(yards, "p90"),
            "positive_gain_rate": float(np.mean(arr > 0.0)) if events else None,
            "explosive_15_rate": float(np.mean(arr >= 15.0)) if events else None,
            "explosive_20_rate": float(np.mean(arr >= 20.0)) if events else None,
            "explosive_40_rate": float(np.mean(arr >= 40.0)) if events else None,
            "explosive_15_contribution_per_scrimmage": (
                float(np.sum(arr >= 15.0)) / total if total else 0.0
            ),
            "touchdown_rate": float(np.mean(touchdowns)) if touchdowns else None,
            "air_yards_mean": _stat(bucket.get("air_yards", []), "mean"),
            "air_yards_p90": _stat(bucket.get("air_yards", []), "p90"),
            "yac_mean": _stat(bucket.get("yac", []), "mean"),
            "yac_p90": _stat(bucket.get("yac", []), "p90"),
            "yards_before_contact_mean": _stat(
                bucket.get("yards_before_contact", []), "mean"
            ),
            "yards_after_contact_mean": _stat(
                bucket.get("yards_after_contact", []), "mean"
            ),
        }
        rows.append(row)
    return rows


def _comparison(
    simulated: list[dict[str, object]], historical: list[dict[str, object]]
) -> list[dict[str, object]]:
    hist = {str(row["family"]): row for row in historical}
    metrics = (
        "share_of_scrimmage",
        "yards_mean",
        "yards_p90",
        "explosive_15_rate",
        "explosive_20_rate",
        "explosive_40_rate",
        "explosive_15_contribution_per_scrimmage",
        "touchdown_rate",
        "air_yards_mean",
        "yac_mean",
    )
    rows = []
    for sim_row in simulated:
        family = str(sim_row["family"])
        hist_row = hist[family]
        for metric in metrics:
            sim_value = sim_row.get(metric)
            hist_value = hist_row.get(metric)
            if sim_value is None or hist_value is None:
                continue
            simulated_value = float(sim_value)
            historical_value = float(hist_value)
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "simulated": simulated_value,
                    "historical": historical_value,
                    "absolute_delta": simulated_value - historical_value,
                    "relative_delta": (
                        (simulated_value - historical_value) / historical_value
                        if abs(historical_value) > 1e-12
                        else None
                    ),
                }
            )
    return rows


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
    parser.add_argument("--out", type=Path, default=Path("artifacts/play-gain-reality"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
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

    simulated_store: dict[str, defaultdict[str, list[float]]] = {}
    simulated_by_game: dict[str, dict[str, defaultdict[str, list[float]]]] = {}
    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        game_store: dict[str, defaultdict[str, list[float]]] = {}
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
                _append_sim_event(simulated_store, event)
                _append_sim_event(game_store, event)

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)
    historical_store: dict[str, defaultdict[str, list[float]]] = {}
    for row in pbp.to_dicts():
        _append_historical_row(historical_store, row)

    simulated = _summaries(
        simulated_store, source="monster_v1_3", scope="2026_week1_12_game_slate"
    )
    historical = _summaries(
        historical_store, source="nflverse", scope=f"{args.history}_regular_season"
    )
    by_game = [
        row
        for game, store in simulated_by_game.items()
        for row in _summaries(store, source="monster_v1_3", scope=game)
    ]
    comparison = _comparison(simulated, historical)

    comp_by_key = {(row["family"], row["metric"]): row for row in comparison}
    diagnosis = {
        "all_scrimmage_explosive_15_relative_delta": comp_by_key[
            ("all_scrimmage", "explosive_15_rate")
        ]["relative_delta"],
        "designed_run_explosive_15_relative_delta": comp_by_key[
            ("designed_run", "explosive_15_rate")
        ]["relative_delta"],
        "scramble_explosive_15_relative_delta": comp_by_key[
            ("scramble", "explosive_15_rate")
        ]["relative_delta"],
        "complete_pass_explosive_15_relative_delta": comp_by_key[
            ("complete_pass", "explosive_15_rate")
        ]["relative_delta"],
        "complete_pass_air_yards_relative_delta": comp_by_key.get(
            ("complete_pass", "air_yards_mean"), {}
        ).get("relative_delta"),
        "complete_pass_yac_relative_delta": comp_by_key.get(
            ("complete_pass", "yac_mean"), {}
        ).get("relative_delta"),
        "principle": "This audit localizes gain-generation mismatch by causal play family. It does not authorize probability or efficiency tuning.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(simulated).write_csv(args.out / "simulated_play_gain_summary.csv")
    pl.DataFrame(historical).write_csv(args.out / "historical_play_gain_summary.csv")
    pl.DataFrame(by_game).write_csv(args.out / "simulated_play_gain_by_game.csv")
    pl.DataFrame(comparison).write_csv(args.out / "play_gain_comparison.csv")
    manifest = {
        "artifact": "Monster v1.3 Play-Family Gain Reality Audit",
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
        "definitions": {
            "designed_run": "Monster RUN; nflverse rush_attempt == 1 and qb_dropback != 1",
            "scramble": "Monster PASS->SCRAMBLE; nflverse qb_scramble == 1",
            "complete_pass": "Monster PASS->COMPLETE; nflverse complete_pass == 1 after sack/scramble/interception exclusions",
            "explosive": "credited scrimmage yards >= 15",
            "air_yards_yac": "reported only for completed passes; missing historical values are excluded from those component means",
        },
        "diagnosis": diagnosis,
        "files": {
            "simulated": "simulated_play_gain_summary.csv",
            "historical": "historical_play_gain_summary.csv",
            "by_game": "simulated_play_gain_by_game.csv",
            "comparison": "play_gain_comparison.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
