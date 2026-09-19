from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from monster.dfs.optimizer_milp import solve_world_optimal_milp


ALIASES = {
    "marquisebrown": "hollywoodbrown",
    "joshpalmer": "joshuapalmer",
}


def _norm(value: str) -> str:
    value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
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
            pl.col("Team").cast(pl.String).alias("team_id"),
            pl.col("Position").cast(pl.String).alias("position"),
        )
    required = {"Player", "FanDuel_ID", "Salary", "Team", "Position"}
    missing = required - set(fd.columns)
    if missing:
        raise SystemExit(f"FanDuel salary file missing columns: {sorted(missing)}")
    return fd.select(
        pl.col("Player").alias("player"),
        pl.col("FanDuel_ID").cast(pl.String).alias("fanduel_id"),
        pl.col("Salary").cast(pl.Int64).alias("salary"),
        pl.col("Team").cast(pl.String).alias("team_id"),
        pl.col("Position").cast(pl.String).alias("position"),
    )


def _teams_for_game(game: str) -> set[str]:
    away, home = str(game).split("@", 1)
    return {away, home}


def _build_pool(
    worlds: pl.DataFrame,
    fanduel: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    fd_rows = fanduel.with_columns(
        pl.col("position").str.to_uppercase(),
        pl.col("team_id").str.to_uppercase(),
        pl.col("player")
        .map_elements(_norm, return_dtype=pl.String)
        .alias("join_name"),
    ).to_dicts()

    identities = (
        worlds.select("game", "player_id", "player", "position")
        .unique()
        .sort(["game", "position", "player"])
    )

    pool_rows: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for row in identities.to_dicts():
        game = str(row["game"])
        position = str(row["position"]).upper()
        player = str(row["player"])
        player_id = str(row["player_id"])
        game_teams = _teams_for_game(game)

        if position in {"D", "DEF", "DST"}:
            team = (
                player_id.removeprefix("DST_")
                if player_id.startswith("DST_")
                else player.split()[0]
            )
            candidates = [
                fd
                for fd in fd_rows
                if str(fd["team_id"]).upper() == team.upper()
                and str(fd["position"]).upper() in {"D", "DEF", "DST"}
            ]
        else:
            join_name = _norm(player)
            candidates = [
                fd
                for fd in fd_rows
                if str(fd["join_name"]) == join_name
                and str(fd["position"]).upper() == position
                and str(fd["team_id"]).upper() in game_teams
            ]

        if len(candidates) != 1:
            unmatched.append(
                {
                    "game": game,
                    "player_id": player_id,
                    "player": player,
                    "position": position,
                    "candidate_matches": len(candidates),
                }
            )
            continue

        fd = candidates[0]
        pool_rows.append(
            {
                "game": game,
                "player_id": player_id,
                "player": player,
                "team_id": str(fd["team_id"]).upper(),
                "position": (
                    "D" if position in {"D", "DEF", "DST"} else position
                ),
                "salary": int(fd["salary"]),
                "fanduel_id": str(fd["fanduel_id"]),
            }
        )

    pool = pl.DataFrame(pool_rows).sort(
        ["position", "salary", "player"],
        descending=[False, True, False],
    )
    return pool, unmatched


def _world_score_matrix(
    worlds: pl.DataFrame,
    pool: pl.DataFrame,
) -> tuple[np.ndarray, list[int]]:
    world_ids = sorted(int(x) for x in worlds["world"].unique().to_list())
    row_index = {
        (str(row["game"]), str(row["player_id"])): idx
        for idx, row in enumerate(pool.to_dicts())
    }
    matrix = np.zeros((pool.height, len(world_ids)), dtype=np.float32)
    world_col = {world: idx for idx, world in enumerate(world_ids)}

    for row in worlds.select(
        "game", "world", "player_id", "fanduel_points"
    ).iter_rows(named=True):
        key = (str(row["game"]), str(row["player_id"]))
        player_idx = row_index.get(key)
        if player_idx is None:
            continue
        matrix[player_idx, world_col[int(row["world"])]] = float(
            row["fanduel_points"]
        )

    return matrix, world_ids


def _slot_assignment(lineup: pl.DataFrame) -> dict[str, str]:
    rows = lineup.to_dicts()
    by_position: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_position.setdefault(str(row["position"]).upper(), []).append(row)
    for values in by_position.values():
        values.sort(key=lambda row: (-int(row["salary"]), str(row["player"])))

    qb = by_position["QB"]
    rb = by_position["RB"]
    wr = by_position["WR"]
    te = by_position["TE"]
    defense = by_position["D"]

    if len(qb) != 1 or len(defense) != 1:
        raise RuntimeError("Expected exactly one QB and one D/ST in legal lineup")

    flex: dict[str, object] | None = None
    if len(rb) == 3:
        flex = rb.pop()
    elif len(wr) == 4:
        flex = wr.pop()
    elif len(te) == 2:
        flex = te.pop()
    if flex is None:
        raise RuntimeError("Could not identify FanDuel FLEX player")

    return {
        "QB": str(qb[0]["fanduel_id"]),
        "RB1": str(rb[0]["fanduel_id"]),
        "RB2": str(rb[1]["fanduel_id"]),
        "WR1": str(wr[0]["fanduel_id"]),
        "WR2": str(wr[1]["fanduel_id"]),
        "WR3": str(wr[2]["fanduel_id"]),
        "TE": str(te[0]["fanduel_id"]),
        "FLEX": str(flex["fanduel_id"]),
        "D": str(defense[0]["fanduel_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--fanduel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-worlds", type=int, default=None)
    args = parser.parse_args()

    worlds = pl.read_csv(args.worlds)
    required = {
        "game",
        "world",
        "player_id",
        "player",
        "position",
        "fanduel_points",
    }
    missing = required - set(worlds.columns)
    if missing:
        raise SystemExit(f"MONSTER world matrix missing columns: {sorted(missing)}")

    fanduel = _static_fanduel(args.fanduel)
    pool, unmatched = _build_pool(worlds, fanduel)
    matrix, world_ids = _world_score_matrix(worlds, pool)

    if args.limit_worlds is not None:
        world_ids = world_ids[: args.limit_worlds]
        matrix = matrix[:, : len(world_ids)]

    counts = np.zeros(pool.height, dtype=np.int32)
    world_rows: list[dict[str, object]] = []
    lineup_player_rows: list[dict[str, object]] = []
    lineup_counter: Counter[tuple[str, ...]] = Counter()
    started = time.perf_counter()

    for column, world in enumerate(world_ids):
        result = solve_world_optimal_milp(pool, matrix[:, column])
        selected = pool[list(result.indices)]
        counts[list(result.indices)] += 1

        ordered_ids = tuple(
            sorted(str(value) for value in selected["fanduel_id"].to_list())
        )
        lineup_counter[ordered_ids] += 1
        slots = _slot_assignment(selected)
        world_rows.append(
            {
                "world": world,
                "optimal_score": result.score,
                "salary": result.salary,
                **slots,
            }
        )
        for row in selected.to_dicts():
            lineup_player_rows.append(
                {
                    "world": world,
                    "optimal_score": result.score,
                    "lineup_salary": result.salary,
                    **row,
                    "world_points": float(
                        matrix[pool["fanduel_id"].to_list().index(row["fanduel_id"]), column]
                    ),
                }
            )

    elapsed = time.perf_counter() - started
    n_worlds = len(world_ids)
    olr = pool.with_columns(
        pl.Series("optimal_lineups", counts),
        pl.Series("olr", counts / max(n_worlds, 1)),
    ).sort(["olr", "salary"], descending=[True, True])

    unique_rows = []
    world_frame = pl.DataFrame(world_rows)
    for ids, count in lineup_counter.most_common():
        example = world_frame.filter(
            pl.all_horizontal(
                [
                    pl.col(slot).is_in(list(ids))
                    for slot in ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "D")
                ]
            )
        ).head(1)
        unique_rows.append(
            {
                "lineup_key": "|".join(ids),
                "world_optimal_count": count,
                "world_optimal_rate": count / max(n_worlds, 1),
                "example_world": (
                    int(example["world"][0]) if example.height else None
                ),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    pool.write_csv(args.out / "optimizer_pool.csv")
    olr.write_csv(args.out / "player_olr.csv")
    world_frame.write_csv(args.out / "world_optimal_lineups.csv")
    pl.DataFrame(lineup_player_rows).write_csv(
        args.out / "world_optimal_lineup_players.csv"
    )
    pl.DataFrame(unique_rows).write_csv(args.out / "unique_optimal_lineups.csv")
    pl.DataFrame(unmatched).write_csv(args.out / "unmatched.csv")

    upload_path = args.out / "fanduel_world_optimal_upload.csv"
    with upload_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "D"])
        for row in world_rows:
            writer.writerow(
                [
                    row["QB"],
                    row["RB1"],
                    row["RB2"],
                    row["WR1"],
                    row["WR2"],
                    row["WR3"],
                    row["TE"],
                    row["FLEX"],
                    row["D"],
                ]
            )

    summary = {
        "artifact": "MONSTER FanDuel exact world-optimal lineup bridge",
        "worlds": n_worlds,
        "matched_players": pool.height,
        "unmatched_simulated_players": len(unmatched),
        "unique_world_optimal_lineups": len(lineup_counter),
        "solver": "scipy_milp_exact",
        "elapsed_seconds": elapsed,
        "worlds_per_second": n_worlds / elapsed if elapsed > 0 else None,
        "salary_downstream_only": True,
        "ownership_used": False,
        "football_runtime_changed": False,
        "portfolio_status": (
            "WORLD_OPTIMAL_DIAGNOSTIC_NOT_YET_FINAL_GPP_PORTFOLIO"
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(
        olr.select(
            "position",
            "player",
            "team_id",
            "salary",
            "optimal_lineups",
            "olr",
        ).head(40)
    )


if __name__ == "__main__":
    main()
