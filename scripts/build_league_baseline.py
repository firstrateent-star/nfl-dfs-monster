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
from monster.ingest.madden_schema import legacy_madden_adapter_view
from monster.ingest.madden_skill import attach_madden_skill_traits, madden_skill_coverage
from monster.ingest.nflverse import load_league_personnel_inputs
from monster.tabular import write_csv_safe
from monster.teams import NFL_TEAMS


def _write_if_present(frame: pl.DataFrame, path: Path) -> None:
    if frame.height:
        frame.write_parquet(path, compression="zstd")


def _count_match(snapshot: pl.DataFrame, match_type: str) -> int:
    if "madden_official_match_type" not in snapshot.columns:
        return 0
    return int(snapshot.select((pl.col("madden_official_match_type") == match_type).sum()).item())


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
    snapshot = build_league_personnel_snapshot(
        inputs["current_rosters"], inputs["players"], inputs["historical_snap_counts"],
        inputs["current_snap_counts"], recent_games=args.recent_games,
    )
    snapshot = attach_depth_chart(snapshot, inputs["depth_charts"])
    snapshot = infer_game_day_participation(snapshot, season=args.season)
    snapshot = attach_capability_evidence(
        snapshot, inputs["combine"], inputs["pfr_defense_weekly"], inputs["player_stats_history"]
    )

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
    official_match_count = (
        int(snapshot.select(pl.col("madden_official_match_type").is_not_null().sum()).item())
        if "madden_official_match_type" in snapshot.columns else 0
    )
    if madden_source == "ea_official" and madden_ratings.height and official_match_count == 0:
        raise RuntimeError(
            "Official EA Madden ratings loaded but matched zero current NFL personnel rows; "
            "refusing to continue with silently disconnected player identity."
        )

    adapter_ratings = legacy_madden_adapter_view(madden_ratings)
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
    write_csv_safe(snapshot, args.out / "league_personnel.csv")
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
    match_breakdown = {
        "team_name": _count_match(snapshot, "team_name"),
        "name_position": _count_match(snapshot, "name_position"),
        "unique_name": _count_match(snapshot, "unique_name"),
    }
    manifest = {
        "artifact": "Monster 2026 League Personnel Baseline",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "current_season": args.season,
        "history_seasons": args.history,
        "recent_snap_games": args.recent_games,
        "canonical_team_count": len(NFL_TEAMS),
        "snapshot_team_count": snapshot.get_column("team_id").n_unique(),
        "roster_rows": snapshot.height,
        "players_with_snap_prior": int(snapshot.select((pl.col("snap_games_observed") > 0).sum()).item()),
        "players_changed_team_with_snap_history": int(snapshot.select(pl.col("changed_team_since_snap_history").sum()).item()),
        "ol_roster_rows": ol_rows.height,
        "ol_players_with_snap_prior": int(ol_rows.select((pl.col("snap_games_observed") > 0).sum()).item()),
        "ol_players_with_madden_blocking": int(ol_rows.select(pl.col("madden_pass_block").is_not_null().sum()).item()),
        "players_with_depth_role": int(snapshot.select(pl.col("depth_rank").is_not_null().sum()).item()),
        "no_history_depth_starters": int(snapshot.select(((pl.col("snap_games_observed") == 0) & (pl.col("depth_rank") == 1)).sum()).item()),
        "projected_core_players": int(snapshot.select((pl.col("participation_tier") == "core").sum()).item()),
        "projected_rotation_players": int(snapshot.select((pl.col("participation_tier") == "rotation").sum()).item()),
        "players_with_pfr_defender_history": int(snapshot.select((pl.col("defense_games_observed") > 0).sum()).item()),
        "players_with_def_stat_history": int(snapshot.select((pl.col("def_stat_games_observed") > 0).sum()).item()),
        "teams_with_compiled_unit_effects": unit_effects.height,
        "trait_coverage": traits,
        "madden_defense_special_coverage": madden_defense_special,
        "madden_player_ratings_loaded": bool(madden_ratings.height),
        "madden_source": madden_source,
        "madden_source_player_rows": madden_ratings.height,
        "madden_snapshot_attribute_columns": len(madden_attribute_columns),
        "players_with_full_madden_match": official_match_count,
        "madden_official_match_breakdown": match_breakdown,
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
