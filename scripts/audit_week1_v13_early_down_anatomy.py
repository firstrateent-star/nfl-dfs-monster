from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl

import audit_week1_v13_drive_survival as drive
import run_week1_v13_integrated as integrated
import run_week1_v13_root_cause_experiment as root
from monster.ingest.nflverse import configure_cache
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType
from monster.teams import normalize_team_id


def _float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _team(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return normalize_team_id(str(value))
    except (KeyError, ValueError):
        return str(value)


def _distance_bucket(distance: float) -> str:
    if distance <= 3.0:
        return "short_1_3"
    if distance <= 7.0:
        return "medium_4_7"
    return "long_8_plus"


def _field_bucket(yardline_100: float) -> str:
    if yardline_100 < 20.0:
        return "backed_up_1_19"
    if yardline_100 < 40.0:
        return "own_20_39"
    if yardline_100 < 60.0:
        return "middle_40_59"
    if yardline_100 < 80.0:
        return "plus_40_to_21"
    return "red_zone_20_in"


def _pass_depth(air_yards: float | None) -> str:
    if air_yards is None or not math.isfinite(air_yards):
        return "unknown"
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


def _historical_run_geometry(row: dict[str, Any]) -> str:
    if _float(row.get("qb_sneak")) == 1.0:
        return "qb_sneak"
    location = str(row.get("run_location") or "unknown").lower()
    gap = str(row.get("run_gap") or "unknown").lower()
    if location == "middle":
        return "interior"
    if location in {"left", "right"} and gap in {"guard", "unknown"}:
        return "interior"
    if location == "left" and gap == "tackle":
        return "left_offtackle"
    if location == "right" and gap == "tackle":
        return "right_offtackle"
    if location == "left" and gap == "end":
        return "left_edge"
    if location == "right" and gap == "end":
        return "right_edge"
    return "other"


def _run_outcome(yards: float) -> str:
    if yards < 0.0:
        return "run_negative"
    if yards < 3.0:
        return "run_0_2"
    if yards < 5.0:
        return "run_3_4"
    if yards < 10.0:
        return "run_5_9"
    if yards < 15.0:
        return "run_10_14"
    if yards < 20.0:
        return "run_15_19"
    if yards < 40.0:
        return "run_20_39"
    return "run_40_plus"


def _pass_outcome(
    *,
    yards: float,
    complete: bool,
    sack: bool,
    interception: bool,
    scramble: bool,
) -> str:
    if sack:
        return "pass_sack"
    if interception:
        return "pass_interception"
    if scramble:
        return "pass_scramble"
    if not complete:
        return "pass_incomplete"
    if yards < 0.0:
        return "pass_negative_completion"
    if yards < 5.0:
        return "pass_complete_0_4"
    if yards < 10.0:
        return "pass_complete_5_9"
    if yards < 20.0:
        return "pass_complete_10_19"
    if yards < 40.0:
        return "pass_complete_20_39"
    return "pass_complete_40_plus"


def _historical_inputs(*, season: int, cache_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return accepted early-down NFL snaps plus corrected first-scrimmage drive starts."""
    configure_cache(cache_dir)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    pbp = drive._ensure_columns(
        pbp,
        {
            "game_id": None,
            "fixed_drive": None,
            "posteam": None,
            "defteam": None,
            "down": None,
            "ydstogo": None,
            "yardline_100": None,
            "qb_dropback": 0,
            "rush_attempt": 0,
            "qb_scramble": 0,
            "qb_sneak": 0,
            "complete_pass": 0,
            "interception": 0,
            "sack": 0,
            "yards_gained": 0.0,
            "air_yards": None,
            "yards_after_catch": None,
            "run_location": None,
            "run_gap": None,
            "pass_touchdown": 0,
            "rush_touchdown": 0,
            "passer_player_id": None,
            "receiver_player_id": None,
            "rusher_player_id": None,
        },
    )

    scrimmage = (
        ((pl.col("qb_dropback").fill_null(0).cast(pl.Float64) == 1.0)
         | (pl.col("rush_attempt").fill_null(0).cast(pl.Float64) == 1.0))
        & pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    first_scrimmage = (
        pbp.filter(scrimmage & pl.col("fixed_drive").is_not_null())
        .with_columns(
            (100.0 - pl.col("yardline_100").cast(pl.Float64)).alias("start_yardline_100")
        )
        .group_by(["game_id", "fixed_drive", "posteam"], maintain_order=True)
        .agg(pl.col("start_yardline_100").first())
        .drop_nulls(["start_yardline_100"])
    )

    early = pbp.filter(scrimmage & pl.col("down").is_in([1, 2]))
    rows: list[dict[str, Any]] = []
    for row in early.iter_rows(named=True):
        dropback = _float(row.get("qb_dropback")) == 1.0
        rush = _float(row.get("rush_attempt")) == 1.0
        family = "pass" if dropback else "run"
        if not dropback and not rush:
            continue
        yards = _float(row.get("yards_gained"))
        distance = max(_float(row.get("ydstogo"), 10.0), 0.1)
        yardline = 100.0 - _float(row.get("yardline_100"), 75.0)
        air_raw = row.get("air_yards")
        air_yards = None if air_raw is None else _float(air_raw, float("nan"))
        if air_yards is not None and not math.isfinite(air_yards):
            air_yards = None
        yac_raw = row.get("yards_after_catch")
        yac = None if yac_raw is None else _float(yac_raw, float("nan"))
        if yac is not None and not math.isfinite(yac):
            yac = None
        complete = _float(row.get("complete_pass")) == 1.0
        sack = _float(row.get("sack")) == 1.0
        interception = _float(row.get("interception")) == 1.0
        scramble = _float(row.get("qb_scramble")) == 1.0
        intent = _pass_depth(air_yards) if family == "pass" else _historical_run_geometry(row)
        outcome = (
            _pass_outcome(
                yards=yards,
                complete=complete,
                sack=sack,
                interception=interception,
                scramble=scramble,
            )
            if family == "pass"
            else _run_outcome(yards)
        )
        rows.append(
            {
                "source": "historical",
                "game": str(row.get("game_id") or "unknown"),
                "world": -1,
                "seed": -1,
                "offense_team": _team(row.get("posteam")),
                "defense_team": _team(row.get("defteam")),
                "down": int(_float(row.get("down"), 1.0)),
                "distance": distance,
                "distance_bucket": _distance_bucket(distance),
                "yardline_100": yardline,
                "field_position_bucket": _field_bucket(yardline),
                "play_family": family,
                "outcome_bucket": outcome,
                "yards": yards,
                "air_yards": air_yards,
                "yards_after_catch": yac,
                "yards_before_contact": None,
                "yards_after_contact": None,
                "intent_category": intent,
                "is_completion": complete,
                "is_sack": sack,
                "is_interception": interception,
                "is_scramble": scramble,
                "is_touchdown": (_float(row.get("pass_touchdown")) == 1.0)
                or (_float(row.get("rush_touchdown")) == 1.0),
                "offense_pass_efficiency": None,
                "offense_rush_efficiency": None,
                "quarterback_efficiency": None,
                "defense_pressure_rate": None,
                "defense_run_stuff_rate": None,
                "actor_efficiency": None,
                "actor_explosive": None,
            }
        )
    return pl.DataFrame(rows), first_scrimmage


def _actor_map(team: Any) -> dict[str, Any]:
    players = (team.quarterback, *team.rushers, *team.receivers)
    return {player.player_id: player for player in players}


def _simulated_inputs(
    *,
    pools: dict[str, Any],
    teams: dict[str, Any],
    defenses: dict[str, Any],
    ecology: Any,
    worlds: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, Any]] = []
    drive_starts: list[dict[str, Any]] = []
    config = root.CONFIGS["E_isolated_suppressed"]
    actor_maps = {team_id: _actor_map(team) for team_id, team in teams.items()}
    total_games = len(integrated.MATCHUPS) * worlds
    completed_games = 0
    started = time.monotonic()

    for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
        game = f"{away}@{home}"
        for world in range(worlds):
            world_seed = seed + game_idx * 1_000_003 + world
            away_plan = root.sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(world_seed + 101_003)
            )
            home_plan = root.sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(world_seed + 202_007)
            )
            away_team = integrated._with_event_rush_plan(teams[away], away_plan)
            home_team = integrated._with_event_rush_plan(teams[home], home_plan)
            streams = root.RngStreams.from_seed(world_seed + 7_000_019)
            audit: defaultdict[str, float] = defaultdict(float)
            captured: list[tuple[Any, PlayEvent]] = []
            recorder = root.game_loop.DriveTraceRecorder
            native_observe = recorder.observe

            def observe_with_capture(self, before, event):
                native_observe(self, before, event)
                if before.down in {1, 2} and event.play_type in {PlayType.RUN, PlayType.PASS}:
                    captured.append((before, event))

            recorder.observe = observe_with_capture
            try:
                with root._experiment_runtime(
                    config,
                    streams=streams,
                    ecology=ecology,
                    audit=audit,
                ):
                    result = root.game_loop.simulate_game(
                        away_team,
                        home_team,
                        away_defense=defenses[away],
                        home_defense=defenses[home],
                        seed=world_seed,
                        chaos_ecology=ecology,
                    )
            finally:
                recorder.observe = native_observe

            for trace in result.drive_traces:
                if trace.scrimmage_plays > 0:
                    drive_starts.append(
                        {
                            "source": "monster",
                            "game": game,
                            "world": world,
                            "start_yardline_100": float(trace.start_yardline_100),
                        }
                    )

            for before, event in captured:
                offense = teams[before.possession]
                defense = defenses[before.defense]
                family = "pass" if event.play_type == PlayType.PASS else "run"
                complete = event.pass_result == PassResult.COMPLETE
                sack = event.pass_result == PassResult.SACK
                interception = event.pass_result == PassResult.INTERCEPTION
                scramble = event.pass_result == PassResult.SCRAMBLE
                actor_id = event.target_id if family == "pass" else event.rusher_id
                actor = actor_maps[before.possession].get(actor_id) if actor_id is not None else None
                air_yards = float(event.air_yards) if family == "pass" else None
                intent = (
                    event.pass_depth_category or _pass_depth(air_yards)
                    if family == "pass"
                    else event.run_geometry_category
                    or (event.run_lane.value if event.run_lane is not None else "other")
                )
                outcome = (
                    _pass_outcome(
                        yards=float(event.yards),
                        complete=complete,
                        sack=sack,
                        interception=interception,
                        scramble=scramble,
                    )
                    if family == "pass"
                    else _run_outcome(float(event.yards))
                )
                rows.append(
                    {
                        "source": "monster",
                        "game": game,
                        "world": world,
                        "seed": world_seed,
                        "offense_team": before.possession,
                        "defense_team": before.defense,
                        "down": before.down,
                        "distance": float(before.distance),
                        "distance_bucket": _distance_bucket(float(before.distance)),
                        "yardline_100": float(before.yardline_100),
                        "field_position_bucket": _field_bucket(float(before.yardline_100)),
                        "play_family": family,
                        "outcome_bucket": outcome,
                        "yards": float(event.yards),
                        "air_yards": air_yards,
                        "yards_after_catch": float(event.yards_after_catch) if family == "pass" else None,
                        "yards_before_contact": float(event.yards_before_contact) if family == "run" else None,
                        "yards_after_contact": float(event.yards_after_contact) if family == "run" else None,
                        "intent_category": intent,
                        "is_completion": complete,
                        "is_sack": sack,
                        "is_interception": interception,
                        "is_scramble": scramble,
                        "is_touchdown": bool(event.touchdown),
                        "offense_pass_efficiency": float(offense.pass_efficiency),
                        "offense_rush_efficiency": float(offense.rush_efficiency),
                        "quarterback_efficiency": float(offense.quarterback.efficiency),
                        "defense_pressure_rate": float(defense.pressure_rate),
                        "defense_run_stuff_rate": float(defense.run_stuff_rate),
                        "actor_efficiency": None if actor is None else float(actor.efficiency),
                        "actor_explosive": None if actor is None else float(actor.explosive),
                    }
                )

            completed_games += 1
            if completed_games % 100 == 0 or completed_games == total_games:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed_games / elapsed
                eta = (total_games - completed_games) / rate if rate > 0 else 0.0
                print(
                    f"EARLY-DOWN {completed_games:,}/{total_games:,} games | {game} "
                    f"world {world + 1}/{worlds} | {rate:.2f}/s | ETA {eta / 60:.1f}m",
                    flush=True,
                )

    return pl.DataFrame(rows), pl.DataFrame(drive_starts)


def _distribution(frame: pl.DataFrame, context: list[str]) -> pl.DataFrame:
    keys = [*context, "outcome_bucket"]
    counts = frame.group_by(keys).len().rename({"len": "count"})
    totals = counts.group_by(context).agg(pl.col("count").sum().alias("context_total"))
    return counts.join(totals, on=context, how="left").with_columns(
        (pl.col("count") / pl.col("context_total")).alias("rate")
    )


def _compare_distributions(
    historical: pl.DataFrame,
    monster: pl.DataFrame,
    context: list[str],
) -> pl.DataFrame:
    hist = _distribution(historical, context).rename(
        {"count": "historical_count", "context_total": "historical_context_total", "rate": "historical_rate"}
    )
    sim = _distribution(monster, context).rename(
        {"count": "monster_count", "context_total": "monster_context_total", "rate": "monster_rate"}
    )
    keys = [*context, "outcome_bucket"]
    return (
        hist.join(sim, on=keys, how="full", coalesce=True)
        .with_columns(
            pl.col("historical_count").fill_null(0),
            pl.col("monster_count").fill_null(0),
            pl.col("historical_context_total").fill_null(0),
            pl.col("monster_context_total").fill_null(0),
            pl.col("historical_rate").fill_null(0.0),
            pl.col("monster_rate").fill_null(0.0),
        )
        .with_columns(
            (pl.col("monster_rate") - pl.col("historical_rate")).alias("rate_delta_monster_minus_history"),
            (pl.col("monster_rate") - pl.col("historical_rate")).abs().alias("absolute_rate_gap"),
        )
        .sort([*context, "outcome_bucket"])
    )


def _ratio(frame: pl.DataFrame, predicate: pl.Expr) -> float:
    if frame.height == 0:
        return 0.0
    return float(frame.select(predicate.cast(pl.Float64).mean()).item())


def _metrics(frame: pl.DataFrame) -> dict[str, float]:
    runs = frame.filter(pl.col("play_family") == "run")
    passes = frame.filter(pl.col("play_family") == "pass")
    throws = passes.filter(~pl.col("is_sack") & ~pl.col("is_scramble"))
    completions = throws.filter(pl.col("is_completion"))
    return {
        "early_down_run_yards_per_play": float(runs.get_column("yards").mean()) if runs.height else 0.0,
        "early_down_run_negative_rate": _ratio(runs, pl.col("yards") < 0.0),
        "early_down_run_3plus_rate": _ratio(runs, pl.col("yards") >= 3.0),
        "early_down_run_5plus_rate": _ratio(runs, pl.col("yards") >= 5.0),
        "early_down_run_10plus_rate": _ratio(runs, pl.col("yards") >= 10.0),
        "early_down_run_15plus_rate": _ratio(runs, pl.col("yards") >= 15.0),
        "early_down_run_20plus_rate": _ratio(runs, pl.col("yards") >= 20.0),
        "early_down_run_40plus_rate": _ratio(runs, pl.col("yards") >= 40.0),
        "early_down_pass_dropback_yards_per_play": float(passes.get_column("yards").mean()) if passes.height else 0.0,
        "early_down_pass_sack_rate": _ratio(passes, pl.col("is_sack")),
        "early_down_pass_interception_rate": _ratio(passes, pl.col("is_interception")),
        "early_down_pass_scramble_rate": _ratio(passes, pl.col("is_scramble")),
        "early_down_throw_completion_rate": _ratio(throws, pl.col("is_completion")),
        "early_down_negative_completion_rate": _ratio(completions, pl.col("yards") < 0.0),
        "early_down_completion_yards_mean": float(completions.get_column("yards").mean()) if completions.height else 0.0,
        "early_down_air_yards_mean": float(throws.get_column("air_yards").drop_nulls().mean()) if throws.get_column("air_yards").drop_nulls().len() else 0.0,
        "early_down_yac_mean_completed": float(completions.get_column("yards_after_catch").drop_nulls().mean()) if completions.get_column("yards_after_catch").drop_nulls().len() else 0.0,
        "early_down_pass_5plus_rate": _ratio(passes, pl.col("yards") >= 5.0),
        "early_down_pass_10plus_rate": _ratio(passes, pl.col("yards") >= 10.0),
        "early_down_pass_15plus_rate": _ratio(passes, pl.col("yards") >= 15.0),
        "early_down_pass_20plus_rate": _ratio(passes, pl.col("yards") >= 20.0),
        "early_down_pass_40plus_rate": _ratio(passes, pl.col("yards") >= 40.0),
    }


def _metric_comparison(historical: pl.DataFrame, monster: pl.DataFrame) -> pl.DataFrame:
    hist = _metrics(historical)
    sim = _metrics(monster)
    rows = []
    for metric, historical_value in hist.items():
        monster_value = sim[metric]
        delta = monster_value - historical_value
        relative = delta / abs(historical_value) if abs(historical_value) > 1e-12 else 0.0
        rows.append(
            {
                "metric": metric,
                "historical_value": historical_value,
                "monster_value": monster_value,
                "delta_monster_minus_history": delta,
                "relative_gap": relative,
                "absolute_relative_gap": abs(relative),
            }
        )
    return pl.DataFrame(rows).sort("absolute_relative_gap", descending=True)


def _intent_summary(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(["source", "play_family", "intent_category"])
        .agg(
            pl.len().alias("plays"),
            pl.col("yards").mean().alias("yards_mean"),
            (pl.col("yards") < 0.0).mean().alias("negative_rate"),
            (pl.col("yards") >= 5.0).mean().alias("five_plus_rate"),
            (pl.col("yards") >= 10.0).mean().alias("ten_plus_rate"),
            (pl.col("yards") >= 20.0).mean().alias("twenty_plus_rate"),
            (pl.col("yards") >= 40.0).mean().alias("forty_plus_rate"),
            pl.col("is_touchdown").mean().alias("touchdown_rate"),
        )
        .sort(["play_family", "intent_category", "source"])
    )


def _team_summary(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(["source", "offense_team", "play_family"])
        .agg(
            pl.len().alias("plays"),
            pl.col("yards").mean().alias("yards_mean"),
            (pl.col("yards") >= 5.0).mean().alias("five_plus_rate"),
            (pl.col("yards") >= 10.0).mean().alias("ten_plus_rate"),
            (pl.col("yards") >= 20.0).mean().alias("twenty_plus_rate"),
            (pl.col("yards") >= 40.0).mean().alias("forty_plus_rate"),
            pl.col("offense_pass_efficiency").drop_nulls().mean().alias("offense_pass_efficiency"),
            pl.col("offense_rush_efficiency").drop_nulls().mean().alias("offense_rush_efficiency"),
            pl.col("quarterback_efficiency").drop_nulls().mean().alias("quarterback_efficiency"),
        )
        .sort(["play_family", "source", "offense_team"])
    )


def _drive_start_summary(historical: pl.DataFrame, monster: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for source, frame in (("historical_first_scrimmage", historical), ("monster", monster)):
        values = frame.get_column("start_yardline_100").drop_nulls()
        rows.append(
            {
                "source": source,
                "drives": values.len(),
                "mean_start_yardline_100": float(values.mean()),
                "p10": float(values.quantile(0.10, interpolation="linear")),
                "p50": float(values.quantile(0.50, interpolation="linear")),
                "p90": float(values.quantile(0.90, interpolation="linear")),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026190921)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    historical, historical_drive_starts = _historical_inputs(
        season=args.history,
        cache_dir=args.cache_dir,
    )
    pools, teams, defenses, _states, ecology = drive._build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )
    monster, monster_drive_starts = _simulated_inputs(
        pools=pools,
        teams=teams,
        defenses=defenses,
        ecology=ecology,
        worlds=args.worlds,
        seed=args.seed,
    )

    historical.write_csv(args.out / "historical_early_down_plays.csv")
    monster.write_csv(args.out / "simulated_early_down_plays.csv")

    overall = _compare_distributions(historical, monster, ["play_family"])
    overall.write_csv(args.out / "overall_outcome_distribution.csv")

    context = _compare_distributions(
        historical,
        monster,
        ["play_family", "down", "distance_bucket", "field_position_bucket", "intent_category"],
    )
    context.write_csv(args.out / "context_outcome_distribution.csv")

    metrics = _metric_comparison(historical, monster)
    metrics.write_csv(args.out / "early_down_metric_comparison.csv")

    combined = pl.concat([historical, monster], how="diagonal_relaxed")
    _intent_summary(combined).write_csv(args.out / "intent_outcome_summary.csv")
    _team_summary(combined).write_csv(args.out / "team_early_down_summary.csv")

    start_summary = _drive_start_summary(historical_drive_starts, monster_drive_starts)
    start_summary.write_csv(args.out / "drive_start_field_position_correction.csv")

    manifest = {
        "experiment": "MON-LEDGER-002A",
        "parent_experiment": "MON-LEDGER-002",
        "center": "maximum causal fidelity to observable NFL reality",
        "question": "Which early-down outcome channels create Monster's excess first-series termination hazard?",
        "history_season": args.history,
        "historical_early_down_plays": historical.height,
        "simulated_early_down_plays": monster.height,
        "worlds_per_game": args.worlds,
        "games": len(integrated.MATCHUPS),
        "simulation_branch": "E_isolated_suppressed",
        "field_position_correction": {
            "status": "applied",
            "definition": "historical drive start is the first accepted scrimmage snap, never the kickoff row",
            "prior_MON_LEDGER_002_start_yardline_metric_superseded": True,
        },
        "outcome_contract": {
            "run_buckets": ["negative", "0-2", "3-4", "5-9", "10-14", "15-19", "20-39", "40+"],
            "pass_buckets": ["sack", "interception", "scramble", "incomplete", "negative completion", "0-4", "5-9", "10-19", "20-39", "40+"],
            "conditioning": ["down", "distance", "field position", "pass depth or run geometry", "offense", "defense", "current player/team ability evidence"],
        },
        "largest_metric_gaps": metrics.head(12).to_dicts(),
        "governance": {
            "stage": "LAB",
            "production_promoted": False,
            "football_coefficients_changed_for_experiment": False,
            "market_inputs_used": False,
            "historical_outcomes_used_as_audit_targets_only": True,
        },
        "unknowns": [
            "whether ordinary gains are compressed mainly in runs, passes, or both",
            "whether missing medium gains or missing explosive tails dominate drive survival loss",
            "whether pass depth, YAC, run geometry, or contact yards are the strongest causal boundary",
            "whether the gap is broad league-wide or concentrated by team/context",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(metrics.head(20))
    print(overall)
    print(start_summary)


if __name__ == "__main__":
    main()
