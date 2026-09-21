from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.reality.week2_reality_benchmark_v723 import build_reality_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--detail-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    detail, summary = build_reality_benchmark(
        pl.read_csv(args.projected),
        pl.read_csv(args.actual),
    )
    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail.write_csv(args.detail_out)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
