from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

KEYS = ("game", "team_id", "player_id")
METRICS = (
    "receptions_mean",
    "receiving_yards_mean",
    "receiving_yards_p90",
    "rushing_yards_mean",
    "rushing_yards_p90",
    "fd_points_before_turnover_penalties_mean",
    "fd_points_before_turnover_penalties_p90",
)


def _load(root: Path, profile: str) -> pl.DataFrame:
    path = root / profile / "player_distributions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_csv(path)


def _compare(left: pl.DataFrame, right: pl.DataFrame, label: str) -> dict:
    joined = left.select([*KEYS, "player", "position", *METRICS]).join(
        right.select([*KEYS, *METRICS]), on=list(KEYS), how="inner", suffix="_right"
    )
    out: dict[str, object] = {"comparison": label, "matched_players": joined.height}
    changed = 0
    for metric in METRICS:
        delta = (pl.col(f"{metric}_right") - pl.col(metric)).abs()
        summary = joined.select(
            delta.mean().alias("mean_abs_delta"),
            delta.max().alias("max_abs_delta"),
            (delta > 1e-9).sum().alias("changed_players"),
        ).row(0, named=True)
        out[metric] = summary
        changed = max(changed, int(summary["changed_players"]))
    out["any_player_changed"] = changed > 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    structural = _load(args.root, "structural-current")
    human = _load(args.root, "plus-human")
    madden = _load(args.root, "plus-madden")
    units = _load(args.root, "plus-units")
    full = _load(args.root, "full-reality")
    report = {
        "artifact": "Monster player-jurisdiction ablation audit",
        "principle": "Player-trait families are judged where they have causal jurisdiction, not only by game score.",
        "comparisons": [
            _compare(structural, human, "human_vs_structural"),
            _compare(human, madden, "madden_increment"),
            _compare(madden, units, "unit_increment"),
            _compare(units, full, "environment_increment"),
        ],
    }
    # Human and Madden must measurably alter at least one downstream player distribution.
    report["gate_pass"] = bool(
        report["comparisons"][0]["any_player_changed"]
        and report["comparisons"][1]["any_player_changed"]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
