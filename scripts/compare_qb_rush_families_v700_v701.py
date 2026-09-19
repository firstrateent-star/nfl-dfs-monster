from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_attempt_mean(payload: dict[str, Any], family: str) -> float:
    return float(payload["family_summary"][family]["attempts_mean"])


def _family_yards_mean(payload: dict[str, Any], family: str) -> float:
    return float(payload["family_summary"][family]["yards_mean"])


def _historical_attempt_mean(payload: dict[str, Any], family: str) -> float:
    return float(
        payload["family_summary"][family]["mean_attempts_per_start_game_population"]
    )


def _abs_error(value: float, reference: float) -> float:
    return abs(value - reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control = _read(args.control)
    challenger = _read(args.challenger)
    historical = _read(args.historical)

    historical_designed = _historical_attempt_mean(historical, "designed_non_sneak")
    historical_scramble = _historical_attempt_mean(historical, "scramble")
    historical_competitive_attempts = float(
        historical["competitive_qb_rushing"]["mean_attempts_per_start"]
    )
    historical_competitive_yards = float(
        historical["competitive_qb_rushing"]["mean_yards_per_start"]
    )

    def sim_values(payload: dict[str, Any]) -> dict[str, float]:
        pressure_scramble = _family_attempt_mean(payload, "pressure_scramble")
        coverage_scramble = _family_attempt_mean(payload, "coverage_scramble")
        return {
            "designed_attempts": _family_attempt_mean(
                payload, "designed_non_sneak"
            ),
            "sneak_attempts": _family_attempt_mean(payload, "sneak"),
            "pressure_scramble_attempts": pressure_scramble,
            "coverage_scramble_attempts": coverage_scramble,
            "scramble_attempts": pressure_scramble + coverage_scramble,
            "competitive_attempts": float(
                payload["competitive_qb_rushing"]["attempts_mean"]
            ),
            "competitive_yards": float(
                payload["competitive_qb_rushing"]["yards_mean"]
            ),
            "designed_yards": _family_yards_mean(
                payload, "designed_non_sneak"
            ),
        }

    c = sim_values(control)
    n = sim_values(challenger)
    reference = {
        "designed_attempts": historical_designed,
        "scramble_attempts": historical_scramble,
        "competitive_attempts": historical_competitive_attempts,
        "competitive_yards": historical_competitive_yards,
    }

    metrics: dict[str, dict[str, float]] = {}
    for key, ref in reference.items():
        control_error = _abs_error(c[key], ref)
        challenger_error = _abs_error(n[key], ref)
        metrics[key] = {
            "historical_reference": ref,
            "control": c[key],
            "challenger": n[key],
            "delta": n[key] - c[key],
            "control_abs_error": control_error,
            "challenger_abs_error": challenger_error,
            "abs_error_improvement": control_error - challenger_error,
        }

    payload = {
        "experiment": "v7.0.1-qb-rush-family-ecology",
        "control_architecture": "v7.0.0-participation-first-control",
        "challenger_architecture": "v7.0.1-qb-rush-family-shadow",
        "historical_reference_seasons": historical["seasons"],
        "historical_reference_is_market_blind": bool(historical["market_blind"]),
        "week1_2026_used_for_reference": bool(historical["week1_2026_used"]),
        "metrics": metrics,
        "control_family_detail": c,
        "challenger_family_detail": n,
        "interpretation_rule": (
            "Promotion is not decided by this audit alone. Desired evidence is lower "
            "designed/competitive QB rushing error without compensating scramble inflation "
            "or unrelated football regression."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v701_qb_rush_family_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
