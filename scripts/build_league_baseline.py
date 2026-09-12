from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from monster.feature_compile.capability import attach_capability_evidence, capability_coverage_report
from monster.feature_compile.depth import attach_depth_chart
from monster.feature_compile.league_units import compile_league_unit_effects
from monster.feature_compile.participation_inference import infer_game_day_participation, participation_coverage_report
from monster.feature_compile.trait_inputs import trait_coverage
from monster.ingest.league import build_league_personnel_snapshot, league_coverage_report
from monster.ingest.madden_defense_special import attach_madden_defense_special_traits, madden_defense_special_coverage
from monster.ingest.madden_official import attach_all_madden_attributes, load_official_madden27_player_ratings
from monster.ingest.madden_players import attach_madden_ol_ratings, load_madden27_player_ratings, madden_ol_coverage
from monster.ingest.madden_skill import attach_madden_skill_traits, madden_skill_coverage
from monster.ingest.nflverse import load_league_personnel_inputs
from monster.teams import NFL_TEAMS


def _write_if_present(frame: pl.DataFrame, path: Path) -> None:
    if frame.height:
        frame.write_parquet(path, compression="zstd")


def _csv_safe(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a human-readable CSV view while preserving rich columns in Parquet.

    EA exposes abilities and other metadata as nested values.  CSV cannot represent
    Polars List/Array/Struct/Object columns, so serialize those columns only in the
    convenience CSV.  The canonical parquet retains their native types losslessly.
    """
    nested = {pl.List, pl.Array, pl.Struct, pl.Object}
    expressions: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if dtype.base_type() in nested:
            expressions.append(
                pl.col(name)
                .map_elements(
                    lambda value: json.dumps(value, default=str) if value is not None else None,
                    return_dtype=pl.Utf8,
                )
                .alias(name)
            )
    return frame.with_columns(expressions) if expressions else frame


def _legacy_madden_adapter_view(ratings: pl.DataFrame) -> pl.DataFrame:
    if not ratings.height:
        return ratings
    aliases = {
        "full_name": "madden_player_name", "position": "madden_position", "team_name": "madden_team",
        "speed_rating": "madden_speed", "accel_rating": "madden_acceleration", "acceleration_rating": "madden_acceleration",
        "agility_rating": "madden_agility", "awareness_rating": "madden_awareness", "strength_rating": "madden_strength",
        "catch_rating": "madden_catching", "catching_rating": "madden_catching", "carry_rating": "madden_carrying",
        "carrying_rating": "madden_carrying", "throw_power_rating": "madden_throw_power", "kick_power_rating": "madden_kick_power",
        "kick_acc_rating": "madden_kick_accuracy", "kick_accuracy_rating": "madden_kick_accuracy", "run_block_rating": "madden_run_block",
        "pass_block_rating": "madden_pass_block", "tackle_rating": "madden_tackle", "jump_rating": "madden_jumping",
        "kick_ret_rating": "madden_kick_return", "kick_return_rating": "madden_kick_return", "truck_rating": "madden_trucking",
        "change_of_direction_rating": "madden_change_of_direction", "stiff_arm_rating": "madden_stiff_arm", "spin_move_rating": "madden_spin_move",
        "juke_move_rating": "madden_juke_move", "impact_block_rating": "madden_impact_blocking", "run_block_power_rating": "madden_run_block_power",
        "run_block_finesse_rating": "madden_run_block_finesse", "pass_block_power_rating": "madden_pass_block_power",
        "pass_block_finesse_rating": "madden_pass_block_finesse", "throw_acc_short_rating": "madden_throw_accuracy_short",
        "throw_accuracy_short_rating": "madden_throw_accuracy_short", "throw_acc_mid_rating": "madden_throw_accuracy_mid",
        "throw_accuracy_mid_rating": "madden_throw_accuracy_mid", "throw_acc_deep_rating": "madden_throw_accuracy_deep",
        "throw_accuracy_deep_rating": "madden_throw_accuracy_deep", "throw_on_run_rating": "madden_throw_on_run",
        "play_action_rating": "madden_play_action", "throw_under_pressure_rating": "madden_throw_under_pressure",
        "break_sack_rating": "madden_break_sack", "break_tackle_rating": "madden_break_tackle", "spec_catch_rating": "madden_spectacular_catch",
        "spectacular_catch_rating": "madden_spectacular_catch", "cit_rating": "madden_catch_in_traffic",
        "catch_in_traffic_rating": "madden_catch_in_traffic", "route_run_short_rating": "madden_short_route_running",
        "short_route_running_rating": "madden_short_route_running", "route_run_med_rating": "madden_medium_route_running",
        "medium_route_running_rating": "madden_medium_route_running", "route_run_deep_rating": "madden_deep_route_running",
        "deep_route_running_rating": "madden_deep_route_running", "release_rating": "madden_release", "power_moves_rating": "madden_power_moves",
        "finesse_moves_rating": "madden_finesse_moves", "block_shed_rating": "madden_block_shedding",
        "block_shedding_rating": "madden_block_shedding", "pursuit_rating": "madden_pursuit", "play_rec_rating": "madden_play_recognition",
        "play_recognition_rating": "madden_play_recognition", "man_cover_rating": "madden_man_coverage", "man_coverage_rating": "madden_man_coverage",
        "zone_cover_rating": "madden_zone_coverage", "zone_coverage_rating": "madden_zone_coverage", "press_rating": "madden_press",
        "hit_power_rating": "madden_hit_power", "stamina_rating": "madden_stamina", "injury_rating": "madden_injury",
    }
    expressions = [pl.col(source).alias(target) for target, source in aliases.items() if target not in ratings.columns and source in ratings.columns]
    return ratings.with_columns(expressions) if expressions else ratings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--history", type=int, nargs="+", default=[2025])
    parser.add_argument("--recent-games", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/league-baseline"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = load_league_personnel_inputs(args.history, args.season, args.cache_dir)
    snapshot = build_league_personnel_snapshot(inputs["current_rosters"], inputs["players"], inputs["historical_snap_counts"], inputs["current_snap_counts"], recent_games=args.recent_games)
    snapshot = attach_depth_chart(snapshot, inputs["depth_charts"])
    snapshot = infer_game_day_participation(snapshot, season=args.season)
    snapshot = attach_capability_evidence(snapshot, inputs["combine"], inputs["pfr_defense_weekly"], inputs["player_stats_history"])

    madden_source = "ea_official"
    try:
        madden_ratings = load_official_madden27_player_ratings()
    except (OSError, RuntimeError, ValueError, KeyError, requests.RequestException):
        madden_source = "public_mirror_fallback"
        try:
            madden_ratings = load_madden27_player_ratings()
        except (OSError, RuntimeError, ValueError, requests.RequestException):
            madden_source = "unavailable"
            madden_ratings = pl.DataFrame()

    snapshot = attach_all_madden_attributes(snapshot, madden_ratings)
    adapter_ratings = _legacy_madden_adapter_view(madden_ratings)
    snapshot = attach_madden_ol_ratings(snapshot, adapter_ratings)
    snapshot = attach_madden_skill_traits(snapshot, adapter_ratings)
    snapshot = attach_madden_defense_special_traits(snapshot, adapter_ratings)

    coverage = league_coverage_report(snapshot)
    participation_coverage = participation_coverage_report(snapshot)
    capability_coverage = capability_coverage_report(snapshot)
    madden_coverage = madden_ol_coverage(snapshot)
    madden_skill = madden_skill_coverage(snapshot)
    madden_defense_special = madden_defense_special_coverage(snapshot)
    unit_effects = compile_league_unit_effects(snapshot)
    traits = trait_coverage(snapshot)

    snapshot.write_parquet(args.out / "league_personnel.parquet", compression="zstd")
    _csv_safe(snapshot).write_csv(args.out / "league_personnel.csv")
    coverage.write_csv(args.out / "coverage_by_team.csv")
    participation_coverage.write_csv(args.out / "participation_by_team.csv")
    capability_coverage.write_csv(args.out / "capability_by_team.csv")
    if madden_coverage.height:
        madden_coverage.write_csv(args.out / "madden_ol_coverage_by_team.csv")
    if madden_skill.height:
        madden_skill.write_csv(args.out / "madden_skill_coverage_by_team.csv")
    unit_effects.write_csv(args.out / "team_unit_effects.csv")
    for key in ["injuries", "depth_charts", "combine", "pfr_defense_weekly", "player_stats_history", "current_snap_counts", "ff_playerids"]:
        _write_if_present(inputs[key], args.out / f"raw_{key}.parquet")
    _write_if_present(madden_ratings, args.out / "raw_madden27_player_ratings.parquet")

    ol_rows = snapshot.filter(pl.col("position_group") == "OL")
    madden_attribute_columns = [c for c in snapshot.columns if c.startswith("madden_")]
    manifest = {
        "artifact": "Monster 2026 League Personnel Baseline", "generated_at_utc": datetime.now(UTC).isoformat(),
        "current_season": args.season, "history_seasons": args.history, "recent_snap_games": args.recent_games,
        "canonical_team_count": len(NFL_TEAMS), "snapshot_team_count": snapshot.get_column("team_id").n_unique(), "roster_rows": snapshot.height,
        "players_with_snap_prior": int(snapshot.select((pl.col("snap_games_observed") > 0).sum()).item()),
        "players_changed_team_with_snap_history": int(snapshot.select(pl.col("changed_team_since_snap_history").sum()).item()),
        "ol_roster_rows": ol_rows.height, "ol_players_with_snap_prior": int(ol_rows.select((pl.col("snap_games_observed") > 0).sum()).item()),
        "ol_players_with_madden_blocking": int(ol_rows.select(pl.col("madden_pass_block").is_not_null().sum()).item()),
        "players_with_depth_role": int(snapshot.select(pl.col("depth_rank").is_not_null().sum()).item()),
        "no_history_depth_starters": int(snapshot.select(((pl.col("snap_games_observed") == 0) & (pl.col("depth_rank") == 1)).sum()).item()),
        "projected_core_players": int(snapshot.select((pl.col("participation_tier") == "core").sum()).item()),
        "projected_rotation_players": int(snapshot.select((pl.col("participation_tier") == "rotation").sum()).item()),
        "players_with_pfr_defender_history": int(snapshot.select((pl.col("defense_games_observed") > 0).sum()).item()),
        "players_with_def_stat_history": int(snapshot.select((pl.col("def_stat_games_observed") > 0).sum()).item()),
        "teams_with_compiled_unit_effects": unit_effects.height, "trait_coverage": traits, "madden_defense_special_coverage": madden_defense_special,
        "madden_player_ratings_loaded": bool(madden_ratings.height), "madden_source": madden_source, "madden_source_player_rows": madden_ratings.height,
        "madden_snapshot_attribute_columns": len(madden_attribute_columns),
        "players_with_full_madden_match": int(snapshot.select(pl.col("madden_player_id").is_not_null().sum()).item()) if "madden_player_id" in snapshot.columns else 0,
        "madden_authority": "bounded scouting-style proxy; no direct fantasy authority",
        "rich_madden_metadata_storage": "canonical parquet; nested fields serialized only in convenience CSV",
        "principle": "League baseline is canonical; weekly DFS slates are downstream filters.",
    }
    if "forty" in snapshot.columns:
        manifest["players_with_forty"] = int(snapshot.select(pl.col("forty").is_not_null().sum()).item())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
