from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.reality.qb_rush_audit import (
    classify_qb_rush_rows,
    qb_rush_family_manifest,
    summarize_qb_rush_family_worlds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    snap_path = args.simulation / "same_world_snap_participants_v5.csv"
    if not snap_path.exists():
        raise FileNotFoundError(
            "QB rush-family audit requires same_world_snap_participants_v5.csv"
        )

    out = args.out or args.simulation
    out.mkdir(parents=True, exist_ok=True)

    snaps = pl.read_csv(snap_path, infer_schema_length=10000)
    classified = classify_qb_rush_rows(snaps)
    worlds = summarize_qb_rush_family_worlds(classified)
    worlds.write_csv(out / "qb_rush_family_worlds_v7.csv")

    manifest = qb_rush_family_manifest(worlds)
    (out / "qb_rush_family_manifest_v7.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
