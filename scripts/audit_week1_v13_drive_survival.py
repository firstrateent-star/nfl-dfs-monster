from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl

import run_week1_v13_integrated as integrated
import run_week1_v13_root_cause_experiment as root
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.ingest.nflverse import configure_cache
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _ensure_columns(frame: pl.DataFrame, defaults: dict[str, object]) -> pl.DataFrame:
    missing = [pl.lit(value).alias(name) for name, value in defaults.items() if name not in frame.columns]
    return frame.with_columns(*missing) if missing else frame


def _historical_drive_table(*, season: int, cache_dir: Path) -> pl.DataFrame:
    """Compile NFL drive anatomy in the same coordinate system as Monster.

    This is an audit-only reality view. It does not feed probabilities back into the simulator.
    Monster uses distance traveled from the offense's own goal line, while nflverse yardline_100
    is distance to the opponent goal line, so historical field position is mirrored here.
    """
    configure_cache(cache_dir)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "fixed_drive" not in pbp.columns:
        raise RuntimeError("nflverse play-by-play does not expose fixed_drive")

    pbp = _ensure_columns(
        pbp,
        {
            "qb_dropback": 0,
            "rush_attempt": 0,
            "interception": 0,
            "fumble_lost": 0,
            "pass_touchdown": 0,
            "rush_touchdown": 0,
            "punt_attempt": 0,
            "field_goal_attempt": 0,
            "field_goal_result": None,
            "sack": 0,
            "goal_to_go": 0,
            "yards_gained": 0.0,
            "ydstogo": None,
            "down": None,
            "yardline_100": None,
        },
    )
    pbp = pbp.filter(pl.col("posteam").is_not_null() & pl.col("fixed_drive").is_not_null())
    pbp = pbp.with_columns(
        (100.0 - pl.col("yardline_100").cast(pl.Float64)).alias("monster_yardline_100"),
        pl.col("yards_gained").cast(pl.Float64).fill_null(0.0).alias("audit_yards"),
    )

    dropback = pl.col("qb_dropback").fill_null(0) == 1
    rush_attempt = pl.col("rush_attempt").fill_null(0) == 1
    scrimmage = (dropback | rush_attempt) & pl.col("down").is_not_null()
    run_snap = rush_attempt & ~dropback & pl.col("down").is_not_null()
    pass_snap = dropback & pl.col("down").is_not_null()
    turnover = (pl.col("interception").fill_null(0) == 1) | (
        pl.col("fumble_lost").fill_null(0) == 1
    )
    offensive_td = (pl.col("pass_touchdown").fill_null(0) == 1) | (
        pl.col("rush_touchdown").fill_null(0) == 1
    )
    converted = (
        scrimmage
        & ~turnover
        & ~offensive_td
        & pl.col("ydstogo").is_not_null()
        & (pl.col("audit_yards") >= pl.col("ydstogo").cast(pl.Float64))
    )
    early_down = scrimmage & pl.col("down").is_in([1, 2])
    third_down = scrimmage & (pl.col("down") == 3)
    third_long = third_down & (pl.col("ydstogo").cast(pl.Float64) >= 7.0)
    red_zone_snap = scrimmage & (pl.col("monster_yardline_100") >= 80.0)
    goal_to_go_snap = scrimmage & (pl.col("goal_to_go").fill_null(0) == 1)
    failed_fourth = (
        scrimmage
        & (pl.col("down") == 4)
        & ~converted
        & ~turnover
        & ~offensive_td
        & (pl.col("audit_yards") < pl.col("ydstogo").cast(pl.Float64).fill_null(999.0))
    )

    drives = (
        pbp.group_by(["game_id", "fixed_drive", "posteam"], maintain_order=True)
        .agg(
            scrimmage.cast(pl.Int64).sum().alias("scrimmage_plays"),
            pl.col("monster_yardline_100").drop_nulls().first().alias("start_yardline_100"),
            (scrimmage & (pl.col("audit_yards") >= 15.0)).cast(pl.Int64).sum().alias("explosive_plays"),
            (scrimmage & (pl.col("down") == 1)).cast(pl.Int64).sum().alias("series_started"),
            converted.cast(pl.Int64).sum().alias("series_converted"),
            (scrimmage & (pl.col("down") == 1)).cast(pl.Int64).sum().alias("first_down_snaps"),
            (scrimmage & (pl.col("down") == 2)).cast(pl.Int64).sum().alias("second_down_snaps"),
            third_down.cast(pl.Int64).sum().alias("third_down_snaps"),
            (scrimmage & (pl.col("down") == 4)).cast(pl.Int64).sum().alias("fourth_down_snaps"),
            (converted & (pl.col("down") == 3)).cast(pl.Int64).sum().alias("third_down_conversions"),
            third_long.cast(pl.Int64).sum().alias("third_and_long_snaps"),
            (converted & third_long).cast(pl.Int64).sum().alias("third_and_long_conversions"),
            pl.when(third_down)
            .then(pl.col("ydstogo").cast(pl.Float64).fill_null(0.0))
            .otherwise(0.0)
            .sum()
            .alias("third_down_distance_total"),
            (early_down & (pl.col("audit_yards") >= 5.0)).cast(pl.Int64).sum().alias("early_down_5plus_gains"),
            (early_down & run_snap).cast(pl.Int64).sum().alias("early_down_run_snaps"),
            pl.when(early_down & run_snap).then(pl.col("audit_yards")).otherwise(0.0).sum().alias("early_down_run_yards_total"),
            (early_down & run_snap & (pl.col("audit_yards") < 0.0)).cast(pl.Int64).sum().alias("early_down_run_negative_gains"),
            (early_down & run_snap & (pl.col("audit_yards") >= 3.0)).cast(pl.Int64).sum().alias("early_down_run_3plus_gains"),
            (early_down & run_snap & (pl.col("audit_yards") >= 5.0)).cast(pl.Int64).sum().alias("early_down_run_5plus_gains"),
            (early_down & run_snap & (pl.col("audit_yards") >= 10.0)).cast(pl.Int64).sum().alias("early_down_run_10plus_gains"),
            (early_down & pass_snap).cast(pl.Int64).sum().alias("early_down_pass_snaps"),
            pl.when(early_down & pass_snap).then(pl.col("audit_yards")).otherwise(0.0).sum().alias("early_down_pass_yards_total"),
            (early_down & pass_snap & (pl.col("audit_yards") < 0.0)).cast(pl.Int64).sum().alias("early_down_pass_negative_gains"),
            (early_down & pass_snap & (pl.col("audit_yards") >= 3.0)).cast(pl.Int64).sum().alias("early_down_pass_3plus_gains"),
            (early_down & pass_snap & (pl.col("audit_yards") >= 5.0)).cast(pl.Int64).sum().alias("early_down_pass_5plus_gains"),
            (early_down & pass_snap & (pl.col("audit_yards") >= 10.0)).cast(pl.Int64).sum().alias("early_down_pass_10plus_gains"),
            red_zone_snap.cast(pl.Int64).max().alias("red_zone_snap_seen"),
            goal_to_go_snap.cast(pl.Int64).max().alias("goal_to_go_snap_seen"),
            turnover.cast(pl.Int64).sum().alias("turnovers"),
            (pl.col("sack").fill_null(0) == 1).cast(pl.Int64).sum().alias("sacks"),
            offensive_td.cast(pl.Int64).max().alias("has_offensive_td"),
            (pl.col("field_goal_result").cast(pl.Utf8).str.to_lowercase() == "made").fill_null(False).cast(pl.Int64).max().alias("has_made_fg"),
            (pl.col("field_goal_attempt").fill_null(0) == 1).cast(pl.Int64).max().alias("has_fg_attempt"),
            (pl.col("punt_attempt").fill_null(0) == 1).cast(pl.Int64).max().alias("has_punt"),
            turnover.cast(pl.Int64).max().alias("has_turnover"),
            failed_fourth.cast(pl.Int64).max().alias("has_failed_fourth"),
        )
        .filter(pl.col("scrimmage_plays") > 0)
        .with_columns(
            pl.when(pl.col("has_offensive_td") == 1)
            .then(pl.lit("touchdown"))
            .when(pl.col("has_made_fg") == 1)
            .then(pl.lit("field_goal"))
            .when((pl.col("has_fg_attempt") == 1) & (pl.col("has_made_fg") == 0))
            .then(pl.lit("missed_field_goal"))
            .when(pl.col("has_punt") == 1)
            .then(pl.lit("punt"))
            .when(pl.col("has_turnover") == 1)
            .then(pl.lit("turnover"))
            .when(pl.col("has_failed_fourth") == 1)
            .then(pl.lit("turnover_on_downs"))
            .otherwise(pl.lit("other"))
            .alias("terminal")
        )
    )
    return drives


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _drive_metrics(frame: pl.DataFrame) -> dict[str, float]:
    rows = frame.height
    total = lambda column: float(frame.get_column(column).sum())
    mean = lambda column: float(frame.get_column(column).mean())
    early_down_snaps = total("first_down_snaps") + total("second_down_snaps")
    red_zone_drives = int(frame.filter(pl.col("red_zone_snap_seen").cast(pl.Boolean)).height)
    goal_to_go_drives = int(frame.filter(pl.col("goal_to_go_snap_seen").cast(pl.Boolean)).height)
    td_drives = int(frame.filter(pl.col("terminal") == "touchdown").height)
    red_zone_td = int(
        frame.filter(
            pl.col("red_zone_snap_seen").cast(pl.Boolean) & (pl.col("terminal") == "touchdown")
        ).height
    )
    goal_to_go_td = int(
        frame.filter(
            pl.col("goal_to_go_snap_seen").cast(pl.Boolean) & (pl.col("terminal") == "touchdown")
        ).height
    )
    return {
        "scrimmage_plays_per_drive": mean("scrimmage_plays"),
        "series_conversion_rate": _safe_ratio(total("series_converted"), total("series_started")),
        "third_down_conversion_rate": _safe_ratio(total("third_down_conversions"), total("third_down_snaps")),
        "third_and_long_share": _safe_ratio(total("third_and_long_snaps"), total("third_down_snaps")),
        "third_and_long_conversion_rate": _safe_ratio(total("third_and_long_conversions"), total("third_and_long_snaps")),
        "third_down_distance_mean": _safe_ratio(total("third_down_distance_total"), total("third_down_snaps")),
        "early_down_5plus_rate": _safe_ratio(total("early_down_5plus_gains"), early_down_snaps),
        "early_down_run_yards_per_snap": _safe_ratio(total("early_down_run_yards_total"), total("early_down_run_snaps")),
        "early_down_pass_yards_per_snap": _safe_ratio(total("early_down_pass_yards_total"), total("early_down_pass_snaps")),
        "early_down_run_negative_rate": _safe_ratio(total("early_down_run_negative_gains"), total("early_down_run_snaps")),
        "early_down_pass_negative_rate": _safe_ratio(total("early_down_pass_negative_gains"), total("early_down_pass_snaps")),
        "explosive_plays_per_drive": _safe_ratio(total("explosive_plays"), rows),
        "sacks_per_drive": _safe_ratio(total("sacks"), rows),
        "turnovers_per_drive": _safe_ratio(total("turnovers"), rows),
        "red_zone_snap_drive_rate": _safe_ratio(red_zone_drives, rows),
        "red_zone_td_conversion_rate": _safe_ratio(red_zone_td, red_zone_drives),
        "goal_to_go_snap_drive_rate": _safe_ratio(goal_to_go_drives, rows),
        "goal_to_go_td_conversion_rate": _safe_ratio(goal_to_go_td, goal_to_go_drives),
        "touchdown_drive_rate": _safe_ratio(td_drives, rows),
        "field_goal_drive_rate": _safe_ratio(int(frame.filter(pl.col("terminal") == "field_goal").height), rows),
        "punt_drive_rate": _safe_ratio(int(frame.filter(pl.col("terminal") == "punt").height), rows),
        "turnover_drive_rate": _safe_ratio(int(frame.filter(pl.col("terminal") == "turnover").height), rows),
        "turnover_on_downs_drive_rate": _safe_ratio(int(frame.filter(pl.col("terminal") == "turnover_on_downs").height), rows),
        "start_yardline_100_mean": mean("start_yardline_100"),
    }


def _survival_curve(frame: pl.DataFrame, max_plays: int = 12) -> dict[int, float]:
    denominator = frame.height
    return {
        n: _safe_ratio(int(frame.filter(pl.col("scrimmage_plays") >= n).height), denominator)
        for n in range(1, max_plays + 1)
    }


def _build_current_world_inputs(
    *,
    policy_path: Path,
    personnel_path: Path,
    player_usage_path: Path,
    situation_context_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    policy = integrated._read(policy_path)
    personnel = integrated._read(personnel_path)
    usage = integrated._read(player_usage_path)
    situation_context = integrated._read(situation_context_path)
    league_neutral_pass_rate, situational_pass_rates = integrated._situational_context(situation_context)
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
    )
    reality = compile_player_reality_inputs(personnel, game_date=integrated.GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states: dict[str, Any] = {}
    for away, home in integrated.MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))
    teams = {
        team: enhanced_team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in integrated.MATCHUPS
        for team in pair
    }
    teams = integrated._attach_historical_intent_ecology(teams, player_usage_path.parent)
    defenses = {
        team: enhanced_defensive_unit(units[team])
        for pair in integrated.MATCHUPS
        for team in pair
    }
    ecology = root._load_chaos_ecology(player_usage_path)
    return pools, teams, defenses, states, ecology


def _simulated_drive_table(
    *,
    pools: dict[str, Any],
    teams: dict[str, Any],
    defenses: dict[str, Any],
    ecology: Any,
    worlds: int,
    seed: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    config = root.CONFIGS["E_isolated_suppressed"]
    total_games = len(integrated.MATCHUPS) * worlds
    completed = 0
    started = time.monotonic()
    for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
        game = f"{away}@{home}"
        for world in range(worlds):
            world_seed = seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(world_seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(world_seed + 202_007)
            )
            away_team = integrated._with_event_rush_plan(teams[away], away_plan)
            home_team = integrated._with_event_rush_plan(teams[home], home_plan)
            streams = root.RngStreams.from_seed(world_seed + 7_000_019)
            audit: dict[str, float] = {}
            from collections import defaultdict

            with root._experiment_runtime(
                config,
                streams=streams,
                ecology=ecology,
                audit=defaultdict(float, audit),
            ):
                result = root.game_loop.simulate_game(
                    away_team,
                    home_team,
                    away_defense=defenses[away],
                    home_defense=defenses[home],
                    seed=world_seed,
                    chaos_ecology=ecology,
                )
            for drive_idx, trace in enumerate(result.drive_traces):
                row = asdict(trace)
                row["terminal"] = trace.terminal.value
                row.update(
                    {
                        "game": game,
                        "world": world,
                        "seed": world_seed,
                        "drive_index": drive_idx,
                    }
                )
                rows.append(row)
            completed += 1
            if completed % 100 == 0 or completed == total_games:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed / elapsed
                eta = (total_games - completed) / rate if rate > 0 else 0.0
                print(
                    f"DRIVE-SURVIVAL {completed:,}/{total_games:,} games | {game} "
                    f"world {world + 1}/{worlds} | {rate:.2f}/s | ETA {eta / 60:.1f}m",
                    flush=True,
                )
    return pl.DataFrame(rows).filter(pl.col("scrimmage_plays") > 0)


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

    historical = _historical_drive_table(season=args.history, cache_dir=args.cache_dir)
    pools, teams, defenses, _states, ecology = _build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )
    simulated = _simulated_drive_table(
        pools=pools,
        teams=teams,
        defenses=defenses,
        ecology=ecology,
        worlds=args.worlds,
        seed=args.seed,
    )

    historical.write_csv(args.out / "historical_drive_traces.csv")
    simulated.write_csv(args.out / "simulated_drive_traces.csv")

    history_metrics = _drive_metrics(historical)
    monster_metrics = _drive_metrics(simulated)
    metric_rows = []
    for metric in history_metrics:
        history_value = history_metrics[metric]
        monster_value = monster_metrics[metric]
        delta = monster_value - history_value
        relative = _safe_ratio(delta, abs(history_value)) if abs(history_value) > 1e-12 else 0.0
        metric_rows.append(
            {
                "metric": metric,
                "historical_value": history_value,
                "monster_value": monster_value,
                "delta_monster_minus_history": delta,
                "relative_gap": relative,
                "absolute_relative_gap": abs(relative),
            }
        )
    comparison = pl.DataFrame(metric_rows).sort("absolute_relative_gap", descending=True)
    comparison.write_csv(args.out / "drive_metric_comparison.csv")

    history_survival = _survival_curve(historical)
    monster_survival = _survival_curve(simulated)
    survival = pl.DataFrame(
        [
            {
                "scrimmage_play_number": n,
                "historical_survival_rate": history_survival[n],
                "monster_survival_rate": monster_survival[n],
                "delta_monster_minus_history": monster_survival[n] - history_survival[n],
            }
            for n in history_survival
        ]
    )
    survival.write_csv(args.out / "drive_survival_curve.csv")

    terminal_values = sorted(set(historical.get_column("terminal").to_list()) | set(simulated.get_column("terminal").to_list()))
    terminal_rows = []
    for terminal in terminal_values:
        hist_rate = _safe_ratio(historical.filter(pl.col("terminal") == terminal).height, historical.height)
        sim_rate = _safe_ratio(simulated.filter(pl.col("terminal") == terminal).height, simulated.height)
        terminal_rows.append(
            {
                "terminal": terminal,
                "historical_rate": hist_rate,
                "monster_rate": sim_rate,
                "delta_monster_minus_history": sim_rate - hist_rate,
            }
        )
    pl.DataFrame(terminal_rows).write_csv(args.out / "drive_terminal_comparison.csv")

    causal_order = [
        "early_down_5plus_rate",
        "early_down_run_yards_per_snap",
        "early_down_pass_yards_per_snap",
        "series_conversion_rate",
        "third_down_distance_mean",
        "third_down_conversion_rate",
        "red_zone_snap_drive_rate",
        "red_zone_td_conversion_rate",
        "goal_to_go_td_conversion_rate",
    ]
    ordered = comparison.filter(pl.col("metric").is_in(causal_order)).with_columns(
        pl.col("metric").replace_strict(
            causal_order,
            list(range(len(causal_order))),
            default=len(causal_order),
        ).alias("causal_order")
    ).sort("causal_order")
    ordered.write_csv(args.out / "causal_drive_funnel.csv")

    manifest = {
        "experiment": "MON-LEDGER-002",
        "center": "maximum causal fidelity to observable NFL reality",
        "question": "At what causal stage does a Monster possession stop resembling an NFL possession?",
        "history_season": args.history,
        "historical_drives": historical.height,
        "simulated_drives": simulated.height,
        "worlds_per_game": args.worlds,
        "games": len(integrated.MATCHUPS),
        "simulation_branch": "E_isolated_suppressed",
        "reason_for_branch": "Audit base drive survival with deterministic isolated RNG while rare chaos consequences are neutralized.",
        "largest_metric_gaps": comparison.head(10).to_dicts(),
        "governance": {
            "stage": "LAB",
            "production_promoted": False,
            "football_coefficients_changed_for_experiment": False,
            "market_inputs_used": False,
            "historical_outcomes_used_as_audit_targets_only": True,
        },
        "source_of_truth": {
            "historical_drive_anatomy": f"nflverse regular-season play-by-play {args.history}",
            "simulation_behavior": "GitHub code at workflow commit",
            "current_personnel": str(args.personnel),
            "current_policy": str(args.policy),
        },
        "unknowns": [
            "first causal stage with material drive-survival divergence",
            "whether early-down inefficiency creates excess third-and-long",
            "whether third-down conversion is independently suppressing drive length",
            "whether red-zone entry or red-zone finishing is the dominant scoring loss",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison.head(15))
    print(survival)


if __name__ == "__main__":
    main()
