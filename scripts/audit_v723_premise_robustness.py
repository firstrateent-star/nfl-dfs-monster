from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from monster.reality.premise_robustness_v723 import build_premise_robustness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-world", type=Path, required=True)
    parser.add_argument("--premise", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-regime-worlds", type=int, default=3)
    args = parser.parse_args()

    report = build_premise_robustness(
        pl.read_csv(args.player_world),
        pl.read_csv(args.premise),
        minimum_regime_worlds=max(args.minimum_regime_worlds, 1),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(args.out)


if __name__ == "__main__":
    main()
