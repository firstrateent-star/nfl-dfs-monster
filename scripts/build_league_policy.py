from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from monster.feature_compile.game_flow_policy import compile_game_flow_policy
from monster.feature_compile.offensive_line import compile_historical_ol_outcomes
from monster.feature_compile.play_intent import (
    compile_pass_intent_policy,
    compile_run_intent_policy,
)
from monster.feature_compile.player import compile_player_usage
from monster.feature_compile.situation import compile_situational_pass_context
from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import PBP_COLUMNS, configure_cache
from monster.teams import NFL_TEAMS


def _write(frame: pl.DataFrame, out: Path, stem: str) -> None:
    frame.write_csv(out / f"{stem}.csv")
    frame.write_parquet(out / f"{stem}.parquet", compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, nargs="+", default=[2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/league-policy"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp(args.history)
    pbp = pbp.select([column for column in PBP_COLUMNS if column in pbp.columns])
    raw_rows = pbp.height
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    regular_rows = pbp.height

    policy = compile_team_policy(pbp)
    situational_pass_context = compile_situational_pass_context(pbp)
    game_flow_league, game_flow_team = compile_game_flow_policy(pbp)
    (
        pass_depth_league,
        pass_depth_team,
        pass_depth_qb,
        pass_depth_outcomes,
        target_depth,
    ) = compile_pass_intent_policy(pbp)
    run_geometry_league, run_geometry_team, run_geometry_rusher, run_geometry_outcomes = (
        compile_run_intent_policy(pbp)
    )
    ol_outcomes = compile_historical_ol_outcomes(pbp)
    player_usage = compile_player_usage(pbp)
    canonical = pl.DataFrame({"team_id": list(NFL_TEAMS)})
    policy = canonical.join(policy, on="team_id", how="left").sort("team_id")
    ol_outcomes = canonical.join(ol_outcomes, on="team_id", how="left").sort("team_id")

    args.out.mkdir(parents=True, exist_ok=True)
    _write(policy, args.out, "team_policy")
    _write(situational_pass_context, args.out, "situational_pass_context")
    _write(game_flow_league, args.out, "game_flow_league")
    _write(game_flow_team, args.out, "game_flow_team")
    _write(pass_depth_league, args.out, "pass_depth_league")
    _write(pass_depth_team, args.out, "pass_depth_team")
    _write(pass_depth_qb, args.out, "pass_depth_qb")
    _write(pass_depth_outcomes, args.out, "pass_depth_outcomes")
    _write(target_depth, args.out, "target_depth")
    _write(run_geometry_league, args.out, "run_geometry_league")
    _write(run_geometry_team, args.out, "run_geometry_team")
    _write(run_geometry_rusher, args.out, "run_geometry_rusher")
    _write(run_geometry_outcomes, args.out, "run_geometry_outcomes")
    _write(ol_outcomes, args.out, "offensive_line_outcomes")
    _write(player_usage, args.out, "player_usage")

    missing = int(policy.select(pl.col("games_observed").is_null().sum()).item())
    ol_missing = int(
        ol_outcomes.select(pl.col("historical_pass_protection_signal").is_null().sum()).item()
    )
    manifest = {
        "artifact": "Monster Historical Team Policy Priors",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "history_seasons": args.history,
        "pbp_scope": "REG" if "season_type" in pbp.columns else "provider_default",
        "raw_pbp_rows": raw_rows,
        "policy_pbp_rows": regular_rows,
        "canonical_team_count": len(NFL_TEAMS),
        "compiled_team_count": policy.height,
        "situational_pass_context_rows": situational_pass_context.height,
        "game_flow_league_rows": game_flow_league.height,
        "game_flow_team_rows": game_flow_team.height,
        "game_flow_team_evidence_raw_unshrunk": True,
        "game_flow_runtime_shrinkage_required": True,
        "pass_depth_league_rows": pass_depth_league.height,
        "pass_depth_team_rows": pass_depth_team.height,
        "pass_depth_qb_rows": pass_depth_qb.height,
        "pass_depth_outcome_rows": pass_depth_outcomes.height,
        "target_depth_rows": target_depth.height,
        "run_geometry_league_rows": run_geometry_league.height,
        "run_geometry_team_rows": run_geometry_team.height,
        "run_geometry_rusher_rows": run_geometry_rusher.height,
        "run_geometry_outcome_rows": run_geometry_outcomes.height,
        "intent_ecology_runtime_authority": "shadow_until_oos_and_paired_gates",
        "player_usage_rows": player_usage.height,
        "teams_missing_observed_games": missing,
        "teams_missing_ol_outcome_prior": ol_missing,
        "market_blind": True,
        "principle": (
            "Historical outcomes are priors and audit targets; simulation policy remains "
            "contextual and season-scope aligned. Pass and run intent evidence is compiled "
            "separately from execution success."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
