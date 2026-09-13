from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl


def _participants(value: object) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=12)
    parser.add_argument("--expected-worlds", type=int, default=100)
    args = parser.parse_args()

    out = args.simulation
    required = [
        "manifest.json",
        "same_world_snap_participants_v5.csv",
        "same_world_personnel_package_summary_v5.csv",
        "same_world_defensive_intent_summary_v5.csv",
        "player_world_fanduel.csv",
        "dst_world_fanduel.csv",
        "dfs_world_fanduel_complete.csv",
        "score_anatomy.json",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"Missing v5 release outputs: {missing}")

    manifest = json.loads((out / "manifest.json").read_text())
    flags = [
        "snap_world_v5_active",
        "per_world_snap_participant_telemetry",
        "actual_11v11_personnel_packages_active",
        "alignment_roles_active",
        "defensive_shell_and_rush_plan_active",
        "production_individual_ol_registration_active",
        "stronger_local_run_interaction_active",
        "same_world_fanduel_dst_active",
        "same_world_complete_fanduel_matrix_active",
    ]
    for flag in flags:
        if manifest.get(flag) is not True:
            raise AssertionError(f"v5 manifest gate inactive: {flag}")

    snaps = pl.read_csv(out / "same_world_snap_participants_v5.csv")
    if snaps.is_empty():
        raise AssertionError("v5 snap telemetry is empty")

    bad_snaps: list[dict[str, object]] = []
    for row in snaps.select(
        "game", "world", "snap_index", "offense_participants", "defense_participants"
    ).iter_rows(named=True):
        offense = _participants(row["offense_participants"])
        defense = _participants(row["defense_participants"])
        if (
            len(offense) != 11
            or len(set(offense)) != 11
            or len(defense) != 11
            or len(set(defense)) != 11
        ):
            bad_snaps.append(row)
            if len(bad_snaps) >= 5:
                break
    if bad_snaps:
        raise AssertionError(f"Non-11v11 snap worlds: {bad_snaps}")

    games = sorted(snaps["game"].unique().to_list())
    if len(games) != args.expected_games:
        raise AssertionError(f"Expected {args.expected_games} games, found {len(games)}")
    worlds_by_game = snaps.group_by("game").agg(pl.col("world").n_unique().alias("worlds"))
    if worlds_by_game["worlds"].min() != args.expected_worlds or worlds_by_game["worlds"].max() != args.expected_worlds:
        raise AssertionError(f"Unexpected worlds per game: {worlds_by_game.to_dicts()}")

    dst = pl.read_csv(out / "dst_world_fanduel.csv")
    expected_dst_rows = args.expected_games * args.expected_worlds * 2
    if dst.height != expected_dst_rows:
        raise AssertionError(f"Expected {expected_dst_rows} D/ST rows, found {dst.height}")
    dst_counts = dst.group_by(["game", "world"]).len()
    if dst_counts["len"].min() != 2 or dst_counts["len"].max() != 2:
        raise AssertionError("Every simulated game world must contain exactly two D/ST rows")

    dfs = pl.read_csv(out / "dfs_world_fanduel_complete.csv")
    if dfs.height <= dst.height:
        raise AssertionError("Complete DFS world matrix contains no offensive players")
    if dfs.filter(pl.col("position") == "D").height != dst.height:
        raise AssertionError("Complete DFS world matrix lost D/ST rows")
    if not all(math.isfinite(float(score)) for score in dfs["fanduel_points"].cast(pl.Float64)):
        raise AssertionError("Non-finite FanDuel score found in complete world matrix")

    score_anatomy = json.loads((out / "score_anatomy.json").read_text())
    release = {
        "games": args.expected_games,
        "worlds_per_game": args.expected_worlds,
        "simulated_games": args.expected_games * args.expected_worlds,
        "scrimmage_snap_rows": snaps.height,
        "dst_world_rows": dst.height,
        "dfs_world_rows": dfs.height,
        "all_snaps_exact_11v11": True,
        "same_world_offense_and_dst": True,
        "score_anatomy_model": score_anatomy.get("model", {}),
        "score_anatomy_gaps": score_anatomy.get("gaps", {}),
    }
    (out / "v5_release_gate.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
