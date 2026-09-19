from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from monster.feature_compile.chaos_priors import compile_chaos_ecology
from monster.feature_compile.game_flow_policy import compile_game_flow_policy
from monster.feature_compile.offensive_line import compile_historical_ol_outcomes
from monster.feature_compile.play_intent import (
    compile_pass_intent_policy,
    compile_run_intent_policy,
)
from monster.feature_compile.player import compile_player_usage
from monster.feature_compile.return_geometry_v633 import compile_return_geometry_v633
from monster.feature_compile.situation import compile_situational_pass_context
from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import PBP_COLUMNS, configure_cache
from monster.teams import NFL_TEAMS


def _write(frame: pl.DataFrame, out: Path, stem: str) -> None:
    frame.write_csv(out / f"{stem}.csv")
    frame.write_parquet(out / f"{stem}.parquet", compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Week 2 football priors from full prior-season PBP plus "
            "current-season PBP only through Week 1."
        )
    )
    parser.add_argument("--prior-seasons", type=int, nargs="+", default=[2025])
    parser.add_argument("--current-season", type=int, default=2026)
    parser.add_argument("--through-week", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    seasons = sorted({*args.prior_seasons, args.current_season})
    pbp = nfl.load_pbp(seasons)
    pbp = pbp.select([column for column in PBP_COLUMNS if column in pbp.columns])
    raw_rows = pbp.height
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "season" not in pbp.columns or "week" not in pbp.columns:
        raise ValueError("PBP must include season and week for Week 2 leakage guard")

    pbp = pbp.filter(
        pl.col("season").is_in(args.prior_seasons)
        | (
            (pl.col("season") == args.current_season)
            & (pl.col("week") <= args.through_week)
        )
    )
    allowed_rows = pbp.height
    leaked = pbp.filter(
        (pl.col("season") == args.current_season)
        & (pl.col("week") > args.through_week)
    ).height
    if leaked:
        raise RuntimeError("Week 2 policy contains post-Week-1 current-season PBP")

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
    (
        run_geometry_league,
        run_geometry_team,
        run_geometry_rusher,
        run_geometry_outcomes,
    ) = compile_run_intent_policy(pbp)
    ol_outcomes = compile_historical_ol_outcomes(pbp)
    player_usage = compile_player_usage(pbp)
    chaos_ecology = compile_chaos_ecology(pbp)
    return_geometry = compile_return_geometry_v633(pbp)
    chaos_ecology = pl.concat([chaos_ecology, return_geometry], how="horizontal")

    canonical = pl.DataFrame({"team_id": list(NFL_TEAMS)})
    policy = canonical.join(policy, on="team_id", how="left").sort("team_id")
    ol_outcomes = canonical.join(
        ol_outcomes,
        on="team_id",
        how="left",
    ).sort("team_id")

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
    _write(chaos_ecology, args.out, "chaos_ecology")

    manifest = {
        "artifact": "Monster 2026 Week 2 Leakage-Resistant Football Policy",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "prior_seasons": args.prior_seasons,
        "current_season": args.current_season,
        "current_season_through_week": args.through_week,
        "source_pbp_rows": raw_rows,
        "allowed_pbp_rows": allowed_rows,
        "post_week1_2026_rows_allowed": 0,
        "week1_2026_real_football_included": True,
        "week2_2026_real_football_included": False,
        "market_blind": True,
        "principle": (
            "Week 1 may update football priors for Week 2, but no Week 2 play, "
            "score, market or fantasy result can influence the simulation."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
