from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.feature_compile.offensive_line import compile_historical_ol_outcomes
from monster.feature_compile.player import compile_player_usage
from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import PBP_COLUMNS, configure_cache
from monster.teams import NFL_TEAMS


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
    ol_outcomes = compile_historical_ol_outcomes(pbp)
    player_usage = compile_player_usage(pbp)
    canonical = pl.DataFrame({"team_id": list(NFL_TEAMS)})
    policy = canonical.join(policy, on="team_id", how="left").sort("team_id")
    ol_outcomes = canonical.join(ol_outcomes, on="team_id", how="left").sort("team_id")

    args.out.mkdir(parents=True, exist_ok=True)
    policy.write_csv(args.out / "team_policy.csv")
    policy.write_parquet(args.out / "team_policy.parquet", compression="zstd")
    ol_outcomes.write_csv(args.out / "offensive_line_outcomes.csv")
    ol_outcomes.write_parquet(args.out / "offensive_line_outcomes.parquet", compression="zstd")
    player_usage.write_csv(args.out / "player_usage.csv")
    player_usage.write_parquet(args.out / "player_usage.parquet", compression="zstd")

    missing = int(policy.select(pl.col("games_observed").is_null().sum()).item())
    ol_missing = int(
        ol_outcomes.select(pl.col("historical_pass_protection_signal").is_null().sum()).item()
    )
    manifest = {
        "artifact": "Monster Historical Team Policy Priors",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_seasons": args.history,
        "pbp_scope": "REG" if "season_type" in pbp.columns else "provider_default",
        "raw_pbp_rows": raw_rows,
        "policy_pbp_rows": regular_rows,
        "canonical_team_count": len(NFL_TEAMS),
        "compiled_team_count": policy.height,
        "player_usage_rows": player_usage.height,
        "teams_missing_observed_games": missing,
        "teams_missing_ol_outcome_prior": ol_missing,
        "market_blind": True,
        "principle": "Historical outcomes are priors and audit targets; simulation policy remains contextual and season-scope aligned.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
