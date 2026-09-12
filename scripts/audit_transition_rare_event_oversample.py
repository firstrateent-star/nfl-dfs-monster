from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from audit_state_transition_mirror import _build_current_teams, _monster_transition_rows
from audit_transition_cause_counterfactuals import _enrich
from monster.sim.state_transition_eval import (
    joint_component_probabilities,
    normalized,
    solve_series_survival,
    transition_probabilities,
    transplant_component_transition,
)

TARGET_STATES = ("d1:7_10", "d2:7_10")
RARE_COMPONENTS = ("scramble_conversion", "defensive_penalty_first_down")


def _sample_target_states(
    rows: list[dict[str, object]],
    *,
    per_state: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for state in TARGET_STATES:
        candidates = [row for row in rows if str(row["coarse_state"]) == state]
        if not candidates:
            raise RuntimeError(f"no historical rows found for {state}")
        take = min(per_state, len(candidates))
        order = rng.permutation(len(candidates)).tolist()
        output.extend(candidates[index] for index in order[:take])
    return output


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials)
        / denom
    )
    return max(center - half, 0.0), min(center + half, 1.0)


def _component_evidence(
    historical: list[dict[str, object]],
    oversampled: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for state in TARGET_STATES:
        h_state = [row for row in historical if str(row["coarse_state"]) == state]
        m_state = [row for row in oversampled if str(row["coarse_state"]) == state]
        for component in RARE_COMPONENTS:
            h_count = sum(str(row["component"]) == component for row in h_state)
            m_count = sum(str(row["component"]) == component for row in m_state)
            h_rate = h_count / len(h_state) if h_state else 0.0
            m_rate = m_count / len(m_state) if m_state else 0.0
            h_lo, h_hi = _wilson_interval(h_count, len(h_state))
            m_lo, m_hi = _wilson_interval(m_count, len(m_state))
            output.append(
                {
                    "coarse_state": state,
                    "component": component,
                    "historical_state_samples": len(h_state),
                    "monster_oversample_state_samples": len(m_state),
                    "historical_component_samples": h_count,
                    "monster_oversample_component_samples": m_count,
                    "historical_rate": h_rate,
                    "monster_oversample_rate": m_rate,
                    "monster_minus_historical_rate": m_rate - h_rate,
                    "historical_wilson_low": h_lo,
                    "historical_wilson_high": h_hi,
                    "monster_wilson_low": m_lo,
                    "monster_wilson_high": m_hi,
                    "intervals_overlap": not (m_hi < h_lo or h_hi < m_lo),
                    "evidence_sufficient": h_count >= 20 and m_count >= 50,
                }
            )
    return output


def _joint_for_state(rows: list[dict[str, object]], state: str) -> dict[str, dict[str, float]]:
    return joint_component_probabilities(
        row for row in rows if str(row["coarse_state"]) == state
    )


def _counterfactuals(
    historical: list[dict[str, object]],
    baseline_monster: list[dict[str, object]],
    oversampled: list[dict[str, object]],
    evidence: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    hist_transitions = transition_probabilities(historical)
    mon_transitions = transition_probabilities(baseline_monster)
    start_distribution = normalized(
        Counter(str(row["coarse_state"]) for row in historical if int(row["down"]) == 1)
    )
    historical_survival, _ = solve_series_survival(hist_transitions, start_distribution)
    monster_survival, _ = solve_series_survival(mon_transitions, start_distribution)
    gap = historical_survival - monster_survival

    evidence_map = {
        (str(row["coarse_state"]), str(row["component"])): bool(row["evidence_sufficient"])
        for row in evidence
    }
    hist_joint = {state: _joint_for_state(historical, state) for state in TARGET_STATES}
    base_joint = {state: _joint_for_state(baseline_monster, state) for state in TARGET_STATES}
    over_joint = {state: _joint_for_state(oversampled, state) for state in TARGET_STATES}

    output: list[dict[str, object]] = []
    for component in RARE_COMPONENTS:
        for scope in (*TARGET_STATES, "d1+d2:7_10"):
            states = list(TARGET_STATES) if scope == "d1+d2:7_10" else [scope]
            if not all(evidence_map.get((state, component), False) for state in states):
                continue

            oversample_baseline = {key: dict(value) for key, value in mon_transitions.items()}
            for state in states:
                oversample_baseline[state] = transplant_component_transition(
                    base_joint[state],
                    over_joint[state],
                    component,
                    mode="mass_and_shape",
                )
            oversample_survival, _ = solve_series_survival(
                oversample_baseline, start_distribution
            )

            for mode in ("mass_and_shape", "rate_only", "shape_only"):
                hybrid = {key: dict(value) for key, value in oversample_baseline.items()}
                for state in states:
                    stabilized_transition = transplant_component_transition(
                        base_joint[state],
                        over_joint[state],
                        component,
                        mode="mass_and_shape",
                    )
                    stabilized_joint = {
                        name: dict(distribution)
                        for name, distribution in base_joint[state].items()
                        if name != component
                    }
                    target_mass = sum(over_joint[state].get(component, {}).values())
                    non_target_mass = sum(
                        sum(distribution.values()) for distribution in stabilized_joint.values()
                    )
                    scale = (1.0 - target_mass) / non_target_mass if non_target_mass > 1e-12 else 0.0
                    for name in list(stabilized_joint):
                        stabilized_joint[name] = {
                            next_state: probability * scale
                            for next_state, probability in stabilized_joint[name].items()
                        }
                    stabilized_joint[component] = dict(over_joint[state].get(component, {}))
                    collapsed = {}
                    for distribution in stabilized_joint.values():
                        for next_state, probability in distribution.items():
                            collapsed[next_state] = collapsed.get(next_state, 0.0) + probability
                    if any(
                        abs(collapsed.get(key, 0.0) - stabilized_transition.get(key, 0.0)) > 1e-9
                        for key in set(collapsed) | set(stabilized_transition)
                    ):
                        raise RuntimeError("oversample stabilization joint failed to reproduce transition")
                    hybrid[state] = transplant_component_transition(
                        stabilized_joint,
                        hist_joint[state],
                        component,
                        mode=mode,
                    )
                survival, _ = solve_series_survival(hybrid, start_distribution)
                delta = survival - oversample_survival
                output.append(
                    {
                        "scope": scope,
                        "component": component,
                        "mode": mode,
                        "states_replaced": len(states),
                        "original_monster_series_survival": monster_survival,
                        "oversample_stabilized_monster_survival": oversample_survival,
                        "hybrid_series_survival": survival,
                        "historical_series_survival": historical_survival,
                        "absolute_recovery_vs_stabilized": delta,
                        "recovery_percentage_points": delta * 100.0,
                        "fraction_of_original_gap_recovered": delta / gap if gap > 1e-12 else 0.0,
                    }
                )

    output.sort(key=lambda row: float(row["absolute_recovery_vs_stabilized"]), reverse=True)
    return output, {
        "historical_series_survival": historical_survival,
        "monster_series_survival": monster_survival,
        "series_survival_gap": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--baseline-monster", type=Path, required=True)
    parser.add_argument("--game-flow-league", type=Path, required=True)
    parser.add_argument("--game-flow-team", type=Path, required=True)
    parser.add_argument("--pass-depth-league", type=Path, required=True)
    parser.add_argument("--pass-depth-team", type=Path, required=True)
    parser.add_argument("--pass-depth-qb", type=Path, required=True)
    parser.add_argument("--pass-depth-outcomes", type=Path, required=True)
    parser.add_argument("--target-depth", type=Path, required=True)
    parser.add_argument("--run-geometry-league", type=Path, required=True)
    parser.add_argument("--run-geometry-team", type=Path, required=True)
    parser.add_argument("--run-geometry-rusher", type=Path, required=True)
    parser.add_argument("--run-geometry-outcomes", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--sample-per-state", type=int, default=2500)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026091203)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    historical_raw = pl.read_parquet(args.historical).to_dicts()
    baseline_monster = _enrich(pl.read_parquet(args.baseline_monster).to_dicts())
    historical = _enrich(historical_raw)
    sampled = _sample_target_states(
        historical_raw,
        per_state=args.sample_per_state,
        seed=args.seed,
    )
    teams, defenses, context = _build_current_teams(args)
    oversampled_raw = _monster_transition_rows(
        sampled,
        teams=teams,
        defenses=defenses,
        pools=context["pools"],
        opponents=context["opponents"],
        replicates=args.replicates,
        seed=args.seed + 17_000_003,
    )
    oversampled = _enrich(oversampled_raw)
    evidence = _component_evidence(historical, oversampled)
    counterfactuals, survival = _counterfactuals(
        historical,
        baseline_monster,
        oversampled,
        evidence,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(oversampled).write_parquet(args.out / "rare_event_monster_rows.parquet")
    pl.DataFrame(evidence).write_csv(args.out / "rare_event_evidence.csv")
    pl.DataFrame(counterfactuals).write_csv(args.out / "rare_event_counterfactuals.csv")

    ledger = {
        "experiment": "MON-LEDGER-002C.3-RARE-EVENT-OVERSAMPLE",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "vlourish_runtime": {
            "center": "Resolve the two remaining low-sample ordinary-down unknowns before opening a production mechanism repair.",
            "boundary": "QB scramble conversions and defensive automatic-first-down penalties only, within d1:7_10 and d2:7_10. No coefficient changes and no full games.",
            "unknown": [
                "Are Monster scramble conversions genuinely too rare after adequate sampling?",
                "Are defensive automatic-first-down penalties genuinely too rare after adequate sampling?",
                "If either deficit is real, how much series survival can correcting only that mechanism recover?",
            ],
            "direction": "Increase evidence density only where the previous audit failed its sample threshold, preserving the same matched-state and current-team context.",
            "practice": [
                "Oversample matched d1:7_10 and d2:7_10 states.",
                "Require at least 50 Monster rare-event observations per state/component before causal ranking.",
                "Compare Wilson intervals for observed NFL and Monster rates.",
                "Use oversampled Monster component estimates to stabilize counterfactual attribution.",
            ],
            "evidence": {
                **survival,
                "sample_per_state": args.sample_per_state,
                "replicates": args.replicates,
                "monster_oversample_rows": len(oversampled),
                "rare_event_evidence": evidence,
                "counterfactuals": counterfactuals,
            },
            "reflection": "PENDING_EVIDENCE_REVIEW",
            "expansion": "If the rare mechanisms are material, include them in the first mechanism repair. Otherwise close them and repair the already-supported ordinary gain-shape and sack-frequency roots first.",
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
        "artifact": "Monster Vlourish Rare Event Transition Oversample",
        "experiment": ledger["experiment"],
        "target_states": list(TARGET_STATES),
        "rare_components": list(RARE_COMPONENTS),
        "sample_per_state": args.sample_per_state,
        "replicates": args.replicates,
        "monster_oversample_rows": len(oversampled),
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "production_authority": False,
        "survival": survival,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(pl.DataFrame(evidence))
    print(pl.DataFrame(counterfactuals))


if __name__ == "__main__":
    main()
