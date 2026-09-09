from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _norm(expr: pl.Expr) -> pl.Expr:
    return expr.str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join frozen Monster fantasy distributions to FanDuel salary metadata.")
    parser.add_argument("--monster", type=Path, required=True)
    parser.add_argument("--fanduel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    monster = pl.read_csv(args.monster).with_columns(_norm(pl.col("player")).alias("join_name"))
    fd = pl.read_csv(args.fanduel)
    required = {"Position", "Nickname", "Salary", "Id"}
    missing = sorted(required - set(fd.columns))
    if missing:
        raise SystemExit(f"FanDuel file missing required columns: {missing}")

    fd = fd.with_columns(_norm(pl.col("Nickname")).alias("join_name"))
    keep = [c for c in ["Position", "Nickname", "Salary", "Id", "Team", "Opponent", "Injury Indicator", "Injury Details"] if c in fd.columns]
    fd = fd.select(["join_name", *keep]).unique("join_name", keep="first")
    joined = monster.join(fd, on="join_name", how="left")
    joined = joined.with_columns(
        (pl.col("fd_mean") / (pl.col("Salary") / 1000.0)).alias("mean_per_1k"),
        (pl.col("fd_p90") / (pl.col("Salary") / 1000.0)).alias("p90_per_1k"),
        (pl.col("fd_p95") / (pl.col("Salary") / 1000.0)).alias("p95_per_1k"),
        (pl.col("fd_p99") / (pl.col("Salary") / 1000.0)).alias("p99_per_1k"),
    )
    eligible = joined.filter(pl.col("Salary").is_not_null())
    unmatched = joined.filter(pl.col("Salary").is_null())
    args.out.mkdir(parents=True, exist_ok=True)
    eligible.sort(["position", "mean_per_1k"], descending=[False, True]).write_csv(args.out / "monster_fanduel_value.csv")
    unmatched.select("game", "team_id", "position", "player", "fd_mean").write_csv(args.out / "monster_unmatched_players.csv")
    print({"monster_rows": joined.height, "salary_matched": eligible.height, "unmatched": unmatched.height})
    print(eligible.select("position", "player", "team_id", "Salary", "fd_mean", "fd_p90", "mean_per_1k", "p90_per_1k").sort("mean_per_1k", descending=True).head(30))


if __name__ == "__main__":
    main()
