from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from monster.sim.state_transition_eval import (
    normalized,
    joint_component_probabilities,
    solve_series_survival,
    transition_probabilities,
    transplant_component_transition,
)

TARGET_STATES = ("d1:7_10", "d2:7_10")
TRANSPLANT_MODES = ("mass_and_shape", "rate_only", "shape_only")


def _component(row: dict[str, object]) -> str:
    cause = str(row.get("cause", "other"))
    yards = float(row.get("yards", 0.0) or 0.0)
    converted = bool(row.get("converted", False))
    touchdown = bool(row.get("touchdown", False))

    if cause.startswith("run_"):
        if converted or touchdown or cause in {"run_conversion", "run_touchdown"}:
            return "run_conversion"
        if "fumble" in cause:
            return "run_turnover"
        return "run_useful_5plus" if yards >= 5.0 else "run_short_under5"

    if cause.startswith("completion_"):
        if converted or touchdown or cause in {"completion_conversion", "completion_touchdown"}:
            return "completion_conversion"
        return "completion_useful_5plus" if yards >= 5.0 else "completion_short_under5"

    if cause.startswith("scramble_"):
        return "scramble_conversion" if converted else "scramble_short"
    if cause == "incomplete":
        return "incomplete"
    if cause == "sack":
        return "sack"
    if cause == "interception":
        return "interception"
    if cause == "offensive_penalty":
        return "offensive_penalty"
    if cause == "defensive_penalty_first_down":
        return "defensive_penalty_first_down"
    if cause == "defensive_penalty":
        return "defensive_penalty_no_auto_first"
    if cause == "punt":
        return "punt"
    if cause == "field_goal":
        return "field_goal"
    return cause


def _enrich(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["component"] = _component(row)
        output.append(enriched)
    return output


def _conditional_success(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("converted")) or bool(row.get("touchdown")) for row in rows) / len(rows)


def _mean_yards(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get("yards", 0.0) or 0.0) for row in rows) / len(rows)


def _decomposition(
    historical: list[dict[str, object]],
    monster: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for state in TARGET_STATES:
        h_state = [row for row in historical if str(row["coarse_state"]) == state]
        m_state = [row for row in monster if str(row["coarse_state"]) == state]
        h_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        m_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in h_state:
            h_groups[str(row["component"])].append(row)
        for row in m_state:
            m_groups[str(row["component"])].append(row)

        for component in sorted(set(h_groups) | set(m_groups)):
            h = h_groups.get(component, [])
            m = m_groups.get(component, [])
            h_rate = len(h) / len(h_state) if h_state else 0.0
            m_rate = len(m) / len(m_state) if m_state else 0.0
            h_success = _conditional_success(h)
            m_success = _conditional_success(m)
            output.append(
                {
                    "coarse_state": state,
                    "component": component,
                    "historical_state_samples": len(h_state),
                    "monster_state_samples": len(m_state),
                    "historical_component_samples": len(h),
                    "monster_component_samples": len(m),
                    "historical_component_rate": h_rate,
                    "monster_component_rate": m_rate,
                    "monster_minus_historical_rate": m_rate - h_rate,
                    "historical_conditional_success": h_success,
                    "monster_conditional_success": m_success,
                    "monster_minus_historical_success": m_success - h_success,
                    "historical_mean_yards": _mean_yards(h),
                    "monster_mean_yards": _mean_yards(m),
                    "monster_minus_historical_mean_yards": _mean_yards(m) - _mean_yards(h),
                    "evidence_sufficient": len(h) >= 20 and len(m) >= 20,
                }
            )
    return output


def _survival_baseline(
    historical: list[dict[str, object]],
    monster: list[dict[str, object]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
    float,
    float,
]:
    hist_transitions = transition_probabilities(historical)
    mon_transitions = transition_probabilities(monster)
    start_distribution = normalized(
        Counter(str(row["coarse_state"]) for row in historical if int(row["down"]) == 1)
    )
    hist_survival, _ = solve_series_survival(hist_transitions, start_distribution)
    mon_survival, _ = solve_series_survival(mon_transitions, start_distribution)
    return hist_transitions, mon_transitions, start_distribution, hist_survival, mon_survival


def _joint_for_state(rows: list[dict[str, object]], state: str) -> dict[str, dict[str, float]]:
    return joint_component_probabilities(
        row for row in rows if str(row["coarse_state"]) == state
    )


def _transplant_rows(
    historical: list[dict[str, object]],
    monster: list[dict[str, object]],
    decomposition: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    _, mon_transitions, start_distribution, hist_survival, mon_survival = _survival_baseline(
        historical, monster
    )
    gap = hist_survival - mon_survival
    evidence = {
        (str(row["coarse_state"]), str(row["component"])): bool(row["evidence_sufficient"])
        for row in decomposition
    }

    historical_joint = {state: _joint_for_state(historical, state) for state in TARGET_STATES}
    monster_joint = {state: _joint_for_state(monster, state) for state in TARGET_STATES}
    components = sorted(
        set().union(
            *(set(historical_joint[state]) | set(monster_joint[state]) for state in TARGET_STATES)
        )
    )

    output: list[dict[str, object]] = []
    for state in TARGET_STATES:
        for component in components:
            if not evidence.get((state, component), False):
                continue
            for mode in TRANSPLANT_MODES:
                hybrid = {key: dict(value) for key, value in mon_transitions.items()}
                hybrid[state] = transplant_component_transition(
                    monster_joint[state],
                    historical_joint[state],
                    component,
                    mode=mode,
                )
                survival, _ = solve_series_survival(hybrid, start_distribution)
                delta = survival - mon_survival
                output.append(
                    {
                        "scope": state,
                        "component": component,
                        "mode": mode,
                        "states_replaced": 1,
                        "monster_baseline_series_survival": mon_survival,
                        "hybrid_series_survival": survival,
                        "historical_series_survival": hist_survival,
                        "absolute_recovery": delta,
                        "recovery_percentage_points": delta * 100.0,
                        "fraction_of_gap_recovered": delta / gap if gap > 1e-12 else 0.0,
                    }
                )

    for component in components:
        if not all(evidence.get((state, component), False) for state in TARGET_STATES):
            continue
        for mode in TRANSPLANT_MODES:
            hybrid = {key: dict(value) for key, value in mon_transitions.items()}
            for state in TARGET_STATES:
                hybrid[state] = transplant_component_transition(
                    monster_joint[state],
                    historical_joint[state],
                    component,
                    mode=mode,
                )
            survival, _ = solve_series_survival(hybrid, start_distribution)
            delta = survival - mon_survival
            output.append(
                {
                    "scope": "d1+d2:7_10",
                    "component": component,
                    "mode": mode,
                    "states_replaced": 2,
                    "monster_baseline_series_survival": mon_survival,
                    "hybrid_series_survival": survival,
                    "historical_series_survival": hist_survival,
                    "absolute_recovery": delta,
                    "recovery_percentage_points": delta * 100.0,
                    "fraction_of_gap_recovered": delta / gap if gap > 1e-12 else 0.0,
                }
            )

    output.sort(key=lambda row: float(row["absolute_recovery"]), reverse=True)
    return output, {
        "historical_series_survival": hist_survival,
        "monster_series_survival": mon_survival,
        "series_survival_gap": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--monster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    historical = _enrich(pl.read_parquet(args.historical).to_dicts())
    monster = _enrich(pl.read_parquet(args.monster).to_dicts())
    decomposition = _decomposition(historical, monster)
    counterfactuals, survival = _transplant_rows(
        historical,
        monster,
        decomposition,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(decomposition).write_csv(args.out / "component_decomposition.csv")
    pl.DataFrame(counterfactuals).write_csv(args.out / "component_counterfactuals.csv")

    mass_shape = [
        row for row in counterfactuals if row["mode"] == "mass_and_shape"
    ]
    rate_only = [row for row in counterfactuals if row["mode"] == "rate_only"]
    shape_only = [row for row in counterfactuals if row["mode"] == "shape_only"]

    ledger = {
        "experiment": "MON-LEDGER-002C.2-CAUSE-COUNTERFACTUALS",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "vlourish_runtime": {
            "center": "Identify which causal outcome mechanisms inside the highest-leverage ordinary first/second-down states create Monster's series-survival loss.",
            "boundary": "Only d1:7_10 and d2:7_10 transition components. Diagnostic transplants only; no football coefficient or production behavior changes.",
            "context": "MON-LEDGER-002C.1 showed d1:7_10 and d2:7_10 carry the highest counterfactual recovery of the series-survival gap.",
            "unknown": [
                "Is the deficit caused primarily by mechanism frequency, conditional outcome shape, or both?",
                "Do ordinary run gains, completions, incompletions, sacks, scrambles, interceptions or penalties have the highest causal leverage?",
                "Does one mechanism explain both first- and second-down loss, suggesting a shared root repair?",
            ],
            "direction": "Decompose each state into mutually exclusive causal components and transplant one component at a time from NFL reality into Monster's transition matrix.",
            "practice": [
                "Measure component rate, conditional success and mean yards.",
                "Run mass+shape counterfactual transplants.",
                "Separate rate-only from shape-only recovery.",
                "Rank repair candidates by recovered series survival and evidence sufficiency.",
            ],
            "evidence": {
                **survival,
                "decomposition_rows": len(decomposition),
                "counterfactual_rows": len(counterfactuals),
                "top_mass_and_shape": mass_shape[:12],
                "top_rate_only": rate_only[:12],
                "top_shape_only": shape_only[:12],
            },
            "reflection": "PENDING_EVIDENCE_REVIEW",
            "expansion": "Open a mechanism-repair experiment only for the component(s) with material, stable counterfactual recovery; validate with a small paired simulation before a 100-world full-slate run.",
        },
        "authority_map": {
            "historical_transition_rows": "authoritative for observed NFL transition frequencies in the 2025 regular-season evidence set",
            "monster_transition_rows": "authoritative for current Stage 3 mechanism behavior on matched historical football states",
            "counterfactual_transplant": "diagnostic attribution only; cannot directly become a production rule",
            "dfs_market": "prohibited upstream",
        },
        "governance": {
            "stage": "LAB",
            "market_blind": True,
            "football_coefficients_changed": False,
            "production_behavior_changed": False,
            "full_game_simulation_required": False,
            "promotion_allowed": False,
        },
    }
    (args.out / "methodology_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    manifest = {
        "artifact": "Monster Vlourish Transition Cause Counterfactual Audit",
        "experiment": ledger["experiment"],
        "target_states": list(TARGET_STATES),
        "historical_rows": len(historical),
        "monster_rows": len(monster),
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "production_authority": False,
        "survival": survival,
        "files": {
            "decomposition": "component_decomposition.csv",
            "counterfactuals": "component_counterfactuals.csv",
            "ledger": "methodology_ledger.json",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(pl.DataFrame(mass_shape[:15]))


if __name__ == "__main__":
    main()
