from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

from monster.dfs.optimizer_milp import solve_world_optimal_milp

ALIASES = {"marquisebrown": "hollywoodbrown", "joshpalmer": "joshuapalmer"}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    for token in ("jr", "sr", "ii", "iii", "iv", "v"):
        value = value.replace(f" {token}.", "").replace(f" {token}", "")
    value = "".join(ch for ch in value if ch.isalnum())
    return ALIASES.get(value, value)


def _static_fanduel(path: Path) -> pl.DataFrame:
    fd = pl.read_csv(path)
    if {"Nickname", "Id", "Salary", "Team", "Position"} <= set(fd.columns):
        return fd.select(
            pl.col("Nickname").alias("player"),
            pl.col("Id").cast(pl.String).alias("fanduel_id"),
            pl.col("Salary").cast(pl.Int64).alias("salary"),
            pl.col("Team").alias("team_id"),
            pl.col("Position").alias("position"),
        )
    required = {"Player", "FanDuel_ID", "Salary", "Team", "Position"}
    missing = required - set(fd.columns)
    if missing:
        raise SystemExit(f"FanDuel static bridge missing columns: {sorted(missing)}")
    return fd.select(
        pl.col("Player").alias("player"),
        pl.col("FanDuel_ID").cast(pl.String).alias("fanduel_id"),
        pl.col("Salary").cast(pl.Int64).alias("salary"),
        pl.col("Team").alias("team_id"),
        pl.col("Position").alias("position"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--distributions", type=Path, required=True)
    parser.add_argument("--fanduel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-worlds", type=int, default=None)
    args = parser.parse_args()

    payload = np.load(args.worlds)
    ids = payload["player_ids"].astype(str)
    scores = payload["fd_points"].astype(np.float32)
    distributions = pl.read_csv(args.distributions)
    if distributions.height != len(ids):
        raise SystemExit("Distribution/world row counts disagree")
    by_id = {str(row["player_id"]): row for row in distributions.to_dicts()}
    fd = _static_fanduel(args.fanduel).with_columns(
        pl.struct(["player", "position", "team_id"])
        .map_elements(
            lambda row: row["team_id"]
            if str(row["position"]).upper() in {"D", "DEF", "DST"}
            else _norm(row["player"]),
            return_dtype=pl.String,
        )
        .alias("join_name")
    )
    fd_map = {
        (str(row["team_id"]), str(row["position"]).upper(), str(row["join_name"])): row
        for row in fd.to_dicts()
    }

    pool_rows = []
    score_rows = []
    unmatched = []
    for idx, player_id in enumerate(ids):
        row = by_id[player_id]
        position = str(row["position"]).upper()
        key_name = (
            str(row["team_id"])
            if position in {"D", "DEF", "DST"}
            else _norm(row["player"])
        )
        candidates = [
            value
            for (team, fd_pos, name), value in fd_map.items()
            if team == str(row["team_id"])
            and name == key_name
            and (fd_pos == position or {fd_pos, position} <= {"D", "DEF", "DST"})
        ]
        if len(candidates) != 1:
            unmatched.append(
                {
                    "player_id": player_id,
                    "player": row["player"],
                    "team_id": row["team_id"],
                    "position": position,
                    "matches": len(candidates),
                }
            )
            continue
        meta = candidates[0]
        pool_rows.append(
            {
                "player_id": player_id,
                "player": row["player"],
                "team_id": row["team_id"],
                "position": "D" if position in {"D", "DEF", "DST"} else position,
                "salary": meta["salary"],
                "fanduel_id": meta["fanduel_id"],
            }
        )
        score_rows.append(idx)

    pool = pl.DataFrame(pool_rows)
    matrix = scores[np.asarray(score_rows), :]
    n_worlds = (
        matrix.shape[1]
        if args.limit_worlds is None
        else min(args.limit_worlds, matrix.shape[1])
    )
    counts = np.zeros(pool.height, dtype=np.int32)
    optimal_scores = np.empty(n_worlds, dtype=np.float32)
    salaries = np.empty(n_worlds, dtype=np.int32)
    started = time.perf_counter()
    for world in range(n_worlds):
        result = solve_world_optimal_milp(pool, matrix[:, world])
        counts[list(result.indices)] += 1
        optimal_scores[world] = result.score
        salaries[world] = result.salary
    elapsed = time.perf_counter() - started

    out = pool.with_columns(
        pl.Series("optimal_lineups", counts), pl.Series("olr", counts / n_worlds)
    ).sort("olr", descending=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out / "player_olr.csv")
    pl.DataFrame(
        {"world": np.arange(n_worlds), "optimal_score": optimal_scores, "salary_used": salaries}
    ).write_csv(args.out / "world_optima.csv")
    pl.DataFrame(unmatched).write_csv(args.out / "unmatched.csv")
    summary = {
        "worlds": n_worlds,
        "matched_players": pool.height,
        "unmatched_players": len(unmatched),
        "defenses": pool.filter(pl.col("position") == "D").height,
        "solver": "scipy_milp_exact",
        "elapsed_seconds": elapsed,
        "worlds_per_second": n_worlds / elapsed,
        "optimal_score_mean": float(optimal_scores.mean()),
        "optimal_score_p90": float(np.quantile(optimal_scores, 0.90)),
        "salary_mean": float(salaries.mean()),
        "salary_min": int(salaries.min()),
        "salary_max": int(salaries.max()),
        "market_blind_football": True,
        "salary_downstream_only": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(out.select("position", "player", "team_id", "salary", "olr").head(30))


if __name__ == "__main__":
    main()
