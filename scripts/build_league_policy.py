from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import configure_cache
from monster.teams import NFL_TEAMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, nargs="+", default=[2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/league-policy"))
    args = parser.parse_args()

    # Keep this heavier PBP-derived artifact separate from the cheap roster refresh.
    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    pbp = nfl.load_pbp(args.history)
    policy = compile_team_policy(pbp)
    canonical = pl.DataFrame({"team_id": list(NFL_TEAMS)})
    policy = canonical.join(policy, on="team_id", how="left").sort("team_id")

    args.out.mkdir(parents=True, exist_ok=True)
    policy.write_csv(args.out / "team_policy.csv")
    policy.write_parquet(args.out / "team_policy.parquet", compression="zstd")

    missing = int(policy.select(pl.col("games_observed").is_null().sum()).item())
    manifest = {
        "artifact": "Monster Historical Team Policy Priors",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_seasons": args.history,
        "canonical_team_count": len(NFL_TEAMS),
        "compiled_team_count": policy.height,
        "teams_missing_observed_games": missing,
        "market_blind": True,
        "principle": "Historical outcomes are priors and audit targets; simulation policy remains contextual.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
