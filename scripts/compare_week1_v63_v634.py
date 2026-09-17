from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


LOWER_IS_BETTER = (
    ("game", "team_points_mae"),
    ("game", "game_total_mae"),
    ("game", "margin_mae"),
    ("game", "total_crps_mean"),
    ("game", "margin_crps_mean"),
    ("player", "fanduel_mae"),
    ("player", "fanduel_rmse"),
    ("player", "fanduel_crps_mean"),
    ("role", "rush_active_player_share_mae"),
    ("role", "target_active_player_share_mae"),
    ("role", "rush_team_share_tvd_mean"),
    ("role", "target_team_share_tvd_mean"),
)

HIGHER_IS_BETTER = (
    ("game", "winner_accuracy"),
    ("player", "fanduel_p10_p90_coverage"),
    ("player", "fanduel_p05_p95_coverage"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict, section: str, key: str) -> float | None:
    value = payload.get(section, {}).get(key)
    return None if value is None else float(value)


def _player_tail_counts(path: Path) -> dict[str, float]:
    frame = pl.read_csv(path)
    n = max(frame.height, 1)
    above95 = frame.filter(pl.col("actual_fanduel") > pl.col("sim_fanduel_p95")).height
    below05 = frame.filter(pl.col("actual_fanduel") < pl.col("sim_fanduel_p05")).height
    return {
        "players": frame.height,
        "above_p95": above95,
        "above_p95_rate": above95 / n,
        "below_p05": below05,
        "below_p05_rate": below05 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    control = _load(args.control / "week1_reality_summary.json")
    challenger = _load(args.challenger / "week1_reality_summary.json")

    metrics: list[dict[str, object]] = []
    for direction, definitions in (
        ("lower", LOWER_IS_BETTER),
        ("higher", HIGHER_IS_BETTER),
    ):
        for section, key in definitions:
            old = _metric(control, section, key)
            new = _metric(challenger, section, key)
            if old is None or new is None:
                continue
            delta = new - old
            improved = delta < 0 if direction == "lower" else delta > 0
            metrics.append(
                {
                    "section": section,
                    "metric": key,
                    "control": old,
                    "challenger": new,
                    "delta": delta,
                    "direction": direction,
                    "improved": improved,
                }
            )

    # These are diagnostics, not optimization targets. In particular, realized Week-1 total SD
    # includes within-game randomness and must never be treated as the desired expected-total SD.
    diagnostics = {}
    for key in (
        "between_matchup_expected_total_sd",
        "mean_within_game_total_sd",
        "expected_total_vs_actual_correlation",
        "game_total_bias",
    ):
        old = _metric(control, "game", key)
        new = _metric(challenger, "game", key)
        diagnostics[key] = {
            "control": old,
            "challenger": new,
            "delta": None if old is None or new is None else new - old,
        }

    control_tail = _player_tail_counts(args.control / "week1_player_reality.csv")
    challenger_tail = _player_tail_counts(args.challenger / "week1_player_reality.csv")

    frame = pl.DataFrame(metrics) if metrics else pl.DataFrame()
    if metrics:
        frame.write_csv(args.out / "paired_reality_metric_deltas.csv")

    report = {
        "experiment": "v6.3.4-paired-week1-reality",
        "control_label": control.get("label"),
        "challenger_label": challenger.get("label"),
        "paired_seed_required": True,
        "metrics": metrics,
        "diagnostics_not_targets": diagnostics,
        "player_tail_calibration": {
            "control": control_tail,
            "challenger": challenger_tail,
            "above_p95_rate_delta": challenger_tail["above_p95_rate"]
            - control_tail["above_p95_rate"],
            "below_p05_rate_delta": challenger_tail["below_p05_rate"]
            - control_tail["below_p05_rate"],
        },
        "guardrails": [
            "Actual Week-1 total SD is not a direct target for between-matchup expected-total SD.",
            "Lower score error alone cannot promote a mechanism.",
            "Promotion requires causal sensitivity, monotonicity, anatomy preservation and paired-seed evidence.",
            "Market or fantasy outcomes are audit-only and never football inputs.",
        ],
        "promotion_status": "SHADOW_EVIDENCE_ONLY",
    }
    (args.out / "v634_week1_paired_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
