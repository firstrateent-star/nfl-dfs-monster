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
DEPTH_BINS = (
    "behind_los",
    "short_0_5",
    "short_6_9",
    "intermediate_10_19",
    "deep_20_39",
    "bomb_40_plus",
)


def _depth_bin(air_yards: float) -> str:
    if air_yards < 0.0:
        return "behind_los"
    if air_yards <= 5.0:
        return "short_0_5"
    if air_yards <= 9.0:
        return "short_6_9"
    if air_yards <= 19.0:
        return "intermediate_10_19"
    if air_yards <= 39.0:
        return "deep_20_39"
    return "bomb_40_plus"


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


def _empty_store() -> defaultdict[str, list[float]]:
    return defaultdict(list)


def _append_throw(
    store: dict[str, defaultdict[str, list[float]]],
    *,
    air_yards: float,
    complete: bool,
    interception: bool,
    touchdown: bool,
    yards: float,
    yac: float | None,
) -> None:
    bucket = store.setdefault(_depth_bin(air_yards), _empty_store())
    bucket["air_yards"].append(float(air_yards))
    bucket["complete"].append(float(complete))
    bucket["interception"].append(float(interception))
    bucket["touchdown"].append(float(touchdown))
    bucket["yards"].append(float(yards))
    if complete:
        bucket["completion_yards"].append(float(yards))
        bucket["negative_completion"].append(float(yards < 0.0))
        bucket["zero_completion"].append(float(yards == 0.0))
        bucket["completion_15_plus"].append(float(yards >= 15.0))
        bucket["completion_20_plus"].append(float(yards >= 20.0))
        bucket["completion_40_plus"].append(float(yards >= 40.0))
        if yac is not None:
            bucket["yac"].append(float(yac))


def _append_sim_event(
    store: dict[str, defaultdict[str, list[float]]], event: PlayEvent
) -> bool:
    if event.play_type != PlayType.PASS:
        return False
    if event.pass_result in {PassResult.SACK, PassResult.SCRAMBLE}:
        return False
    if event.pass_result not in {
        PassResult.COMPLETE,
        PassResult.INCOMPLETE,
        PassResult.INTERCEPTION,
    }:
        return False
    _append_throw(
        store,
        air_yards=float(event.air_yards),
        complete=event.pass_result == PassResult.COMPLETE,
        interception=event.pass_result == PassResult.INTERCEPTION,
        touchdown=bool(event.touchdown),
        yards=float(event.yards),
        yac=float(event.yards_after_catch)
        if event.pass_result == PassResult.COMPLETE
        else None,
    )
    return True


def _historical_is_throw(row: dict[str, Any]) -> bool:
    if _flag(row, "sack") or _flag(row, "qb_scramble"):
        return False
    if "pass_attempt" in row and row.get("pass_attempt") is not None:
        return _flag(row, "pass_attempt")
    return _flag(row, "qb_dropback") and (
        _flag(row, "complete_pass")
        or _flag(row, "incomplete_pass")
        or _flag(row, "interception")
    )


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summaries(
    store: dict[str, defaultdict[str, list[float]]],
    *,
    source: str,
    scope: str,
    known_depth_throws: int,
    all_throws: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in DEPTH_BINS:
        bucket = store.get(depth, _empty_store())
        air = bucket.get("air_yards", [])
        attempts = len(air)
        completions = int(sum(bucket.get("complete", [])))
        completion_yards = bucket.get("completion_yards", [])
        rows.append(
            {
                "source": source,
                "scope": scope,
                "depth_bin": depth,
                "attempts": attempts,
                "attempt_share_known_depth": attempts / known_depth_throws
                if known_depth_throws
                else 0.0,
                "known_depth_throw_coverage": known_depth_throws / all_throws if all_throws else 0.0,
                "air_yards_mean": _mean(air),
                "completion_rate": _mean(bucket.get("complete", [])),
                "interception_rate": _mean(bucket.get("interception", [])),
                "touchdown_rate_per_attempt": _mean(bucket.get("touchdown", [])),
                "yards_per_attempt": _mean(bucket.get("yards", [])),
                "yards_per_completion": _mean(completion_yards),
                "yac_per_completion": _mean(bucket.get("yac", [])),
                "negative_completion_rate": _mean(bucket.get("negative_completion", [])),
                "zero_completion_rate": _mean(bucket.get("zero_completion", [])),
                "completion_15_plus_rate": _mean(bucket.get("completion_15_plus", [])),
                "completion_20_plus_rate": _mean(bucket.get("completion_20_plus", [])),
                "completion_40_plus_rate": _mean(bucket.get("completion_40_plus", [])),
                "completions": completions,
            }
        )
    return rows


def _comparison(
    simulated: list[dict[str, object]], historical: list[dict[str, object]]
) -> list[dict[str, object]]:
    hist = {str(row["depth_bin"]): row for row in historical}
    metrics = (
        "attempt_share_known_depth",
        "air_yards_mean",
        "completion_rate",
        "interception_rate",
        "touchdown_rate_per_attempt",
        "yards_per_attempt",
        "yards_per_completion",
        "yac_per_completion",
        "negative_completion_rate",
        "zero_completion_rate",
        "completion_15_plus_rate",
        "completion_20_plus_rate",
        "completion_40_plus_rate",
    )
    rows: list[dict[str, object]] = []
    for sim_row in simulated:
        depth = str(sim_row["depth_bin"])
        hist_row = hist[depth]
        for metric in metrics:
            sim_value = sim_row.get(metric)
            hist_value = hist_row.get(metric)
            if sim_value is None or hist_value is None:
                continue
            sim_f = float(sim_value)
            hist_f = float(hist_value)
            rows.append(
                {
                    "depth_bin": depth,
                    "metric": metric,
                    "simulated": sim_f,
                    "historical": hist_f,
                    "absolute_delta": sim_f - hist_f,
                    "relative_delta": (sim_f - hist_f) / hist_f
                    if abs(hist_f) > 1e-12
                    else None,
                }
            )
    return rows


def _lookup(
    comparison: list[dict[str, object]], depth: str, metric: str
) -> dict[str, object]:
    for row in comparison:
        if row["depth_bin"] == depth and row["metric"] == metric:
            return row
    raise KeyError((depth, metric))


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
    parser.add_argument("--out", type=Path, default=Path("artifacts/pass-depth-reality"))
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

    simulated_store: dict[str, defaultdict[str, list[float]]] = {}
    sim_known = 0
    sim_all_throws = 0
    for game_idx, (away, home) in enumerate(MATCHUPS):
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
                if event.play_type == PlayType.PASS and event.pass_result not in {
                    PassResult.SACK,
                    PassResult.SCRAMBLE,
                }:
                    sim_all_throws += 1
                if _append_sim_event(simulated_store, event):
                    sim_known += 1

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    historical_store: dict[str, defaultdict[str, list[float]]] = {}
    hist_all_throws = 0
    hist_known = 0
    for row in pbp.to_dicts():
        if not _historical_is_throw(row):
            continue
        hist_all_throws += 1
        value = row.get("air_yards")
        if value is None:
            continue
        try:
            air_yards = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(air_yards):
            continue
        hist_known += 1
        complete = _flag(row, "complete_pass")
        interception = _flag(row, "interception")
        touchdown = _flag(row, "pass_touchdown")
        yac = None
        if complete and row.get("yards_after_catch") is not None:
            yac = _number(row, "yards_after_catch")
        _append_throw(
            historical_store,
            air_yards=air_yards,
            complete=complete,
            interception=interception,
            touchdown=touchdown,
            yards=_number(row, "yards_gained"),
            yac=yac,
        )

    simulated = _summaries(
        simulated_store,
        source="monster_v1_3",
        scope="2026_week1_12_game_slate",
        known_depth_throws=sim_known,
        all_throws=sim_all_throws,
    )
    historical = _summaries(
        historical_store,
        source="nflverse",
        scope=f"{args.history}_regular_season",
        known_depth_throws=hist_known,
        all_throws=hist_all_throws,
    )
    comparison = _comparison(simulated, historical)

    diagnosis = {
        "behind_los_attempt_share_relative_delta": _lookup(
            comparison, "behind_los", "attempt_share_known_depth"
        )["relative_delta"],
        "intermediate_10_19_attempt_share_relative_delta": _lookup(
            comparison, "intermediate_10_19", "attempt_share_known_depth"
        )["relative_delta"],
        "deep_20_39_attempt_share_relative_delta": _lookup(
            comparison, "deep_20_39", "attempt_share_known_depth"
        )["relative_delta"],
        "bomb_40_plus_attempt_share_relative_delta": _lookup(
            comparison, "bomb_40_plus", "attempt_share_known_depth"
        )["relative_delta"],
        "behind_los_negative_completion_rate_delta": _lookup(
            comparison, "behind_los", "negative_completion_rate"
        )["absolute_delta"],
        "intermediate_10_19_completion_rate_relative_delta": _lookup(
            comparison, "intermediate_10_19", "completion_rate"
        )["relative_delta"],
        "deep_20_39_completion_rate_relative_delta": _lookup(
            comparison, "deep_20_39", "completion_rate"
        )["relative_delta"],
        "principle": "Depth is a causal throw-intent/context dimension. This audit diagnoses the shape of the passing distribution; it does not authorize shrinking a single global air-yard mean.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(simulated).write_csv(args.out / "simulated_pass_depth_summary.csv")
    pl.DataFrame(historical).write_csv(args.out / "historical_pass_depth_summary.csv")
    pl.DataFrame(comparison).write_csv(args.out / "pass_depth_comparison.csv")
    manifest = {
        "artifact": "Monster v1.3 Pass Depth Reality Audit",
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
        "simulation_known_depth_coverage": sim_known / sim_all_throws if sim_all_throws else 0.0,
        "historical_known_depth_coverage": hist_known / hist_all_throws if hist_all_throws else 0.0,
        "definitions": {
            "throw": "Monster PASS ending COMPLETE/INCOMPLETE/INTERCEPTION; historical pass attempt excluding sacks and scrambles",
            "behind_los": "air_yards < 0",
            "short_0_5": "0 <= air_yards <= 5",
            "short_6_9": "6 <= air_yards <= 9",
            "intermediate_10_19": "10 <= air_yards <= 19",
            "deep_20_39": "20 <= air_yards <= 39",
            "bomb_40_plus": "air_yards >= 40",
            "attempt_share_denominator": "throws with finite known air_yards",
        },
        "diagnosis": diagnosis,
        "files": {
            "simulated": "simulated_pass_depth_summary.csv",
            "historical": "historical_pass_depth_summary.csv",
            "comparison": "pass_depth_comparison.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
