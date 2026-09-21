from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.dfs.portfolio import PortfolioCandidate, select_failure_path_portfolio

FD_SLOTS = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "D")
DK_SLOTS = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST")


def _seed_to_world(manifest: dict) -> dict[tuple[str, int], int]:
    base_seed = int(manifest["seed"])
    worlds = int(manifest["worlds_per_game"])
    matchups = [str(x) for x in manifest["matchups"]]
    result: dict[tuple[str, int], int] = {}
    for game_idx, game in enumerate(matchups):
        for world in range(worlds):
            seed = base_seed + game_idx * 1_000_003 + world
            result[(game, seed)] = world
    return result


def _scenario_signatures(
    paths: pl.DataFrame,
    manifest: dict,
) -> dict[int, frozenset[str]]:
    lookup = _seed_to_world(manifest)
    signatures: dict[int, set[str]] = {
        world: set()
        for world in range(int(manifest["worlds_per_game"]))
    }
    for row in paths.iter_rows(named=True):
        game = str(row.get("game") or "")
        seed = int(row.get("seed") or -1)
        world = lookup.get((game, seed))
        if world is None:
            continue
        record_type = str(row.get("record_type") or "")
        team = str(row.get("team") or "")
        player = str(row.get("player_id") or "")
        collapse = str(row.get("collapse_mode") or "")
        game_drag = float(row.get("game_drag") or 0.0)
        mutation = str(row.get("mutation") or "")

        if record_type == "world_state":
            if game_drag > 0.0:
                signatures[world].add(f"{game}:GAME_DRAG")
            if collapse and collapse != "none":
                signatures[world].add(
                    f"{game}:{team}:COLLAPSE_{collapse.upper()}"
                )
        elif record_type == "pregame_inactive" and player:
            signatures[world].add(
                f"{game}:{team}:PREGAME_OUT:{player}"
            )
        elif record_type == "live_availability" and player:
            signatures[world].add(
                f"{game}:{team}:LIVE_{mutation.upper()}:{player}"
            )
        elif record_type == "qb_bench":
            signatures[world].add(
                f"{game}:{team}:QB_BENCH:{player}"
            )

    return {
        world: frozenset(values)
        for world, values in signatures.items()
    }


def _slot_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    if set(FD_SLOTS) <= set(frame.columns):
        return FD_SLOTS
    if set(DK_SLOTS) <= set(frame.columns):
        return DK_SLOTS
    raise SystemExit(
        "Could not identify FanDuel or DraftKings lineup slot columns"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineups", type=Path, required=True)
    parser.add_argument("--failure-paths", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--salary-cap", type=int, required=True)
    parser.add_argument("--min-salary", type=int, default=0)
    parser.add_argument("--score-floor-ratio", type=float, default=0.88)
    args = parser.parse_args()

    lineups = pl.read_csv(args.lineups)
    paths = pl.read_csv(args.failure_paths)
    manifest = json.loads(args.manifest.read_text())
    slots = _slot_columns(lineups)
    signatures = _scenario_signatures(paths, manifest)

    by_key: dict[tuple[str, ...], PortfolioCandidate] = {}
    rows_by_key: dict[tuple[str, ...], dict[str, object]] = {}
    for row in lineups.iter_rows(named=True):
        player_ids = tuple(str(row[slot]) for slot in slots)
        key = tuple(sorted(player_ids))
        world = int(row["world"])
        candidate = PortfolioCandidate(
            lineup_id="|".join(key),
            objective_score=float(row["optimal_score"]),
            salary=int(row["salary"]),
            player_ids=player_ids,
            failure_paths=signatures.get(world, frozenset()),
            source_world=world,
        )
        prior = by_key.get(key)
        if prior is None or (
            candidate.objective_score,
            candidate.salary,
        ) > (
            prior.objective_score,
            prior.salary,
        ):
            by_key[key] = candidate
            rows_by_key[key] = row

    selected = select_failure_path_portfolio(
        list(by_key.values()),
        count=args.count,
        salary_cap=args.salary_cap,
        min_salary=args.min_salary,
        score_floor_ratio=args.score_floor_ratio,
    )

    out_rows = []
    for rank, candidate in enumerate(selected, start=1):
        key = tuple(sorted(candidate.player_ids))
        base = dict(rows_by_key[key])
        base.update(
            {
                "portfolio_rank": rank,
                "failure_path_count": len(candidate.failure_paths),
                "failure_path_signature": ";".join(
                    sorted(candidate.failure_paths)
                ),
            }
        )
        out_rows.append(base)

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out_rows).write_csv(
        args.out / "failure_path_portfolio.csv"
    )
    summary = {
        "candidate_lineups": len(by_key),
        "selected_lineups": len(selected),
        "salary_cap": args.salary_cap,
        "min_salary": args.min_salary,
        "score_floor_ratio": args.score_floor_ratio,
        "selection_principle": (
            "ceiling first; failure-path diversity before cosmetic player diversity"
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
