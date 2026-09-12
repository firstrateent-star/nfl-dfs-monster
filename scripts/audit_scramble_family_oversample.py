from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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
)

TARGET_STATES = ("d1:7_10", "d2:7_10")
SCRAMBLE_COMPONENTS = {"scramble_conversion", "scramble_short"}


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


def _sample_state(
    rows: list[dict[str, object]],
    state: str,
    *,
    maximum: int,
    seed: int,
) -> list[dict[str, object]]:
    candidates = [row for row in rows if str(row["coarse_state"]) == state]
    if not candidates:
        raise RuntimeError(f"no historical rows found for {state}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates)).tolist()
    return [candidates[index] for index in order[: min(maximum, len(candidates))]]


def _scramble_rows(rows: list[dict[str, object]], state: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["coarse_state"]) == state
        and str(row.get("component", "")) in SCRAMBLE_COMPONENTS
    ]


def _state_rows(rows: list[dict[str, object]], state: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row["coarse_state"]) == state]


def _evidence(
    historical: list[dict[str, object]],
    oversampled: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for state in TARGET_STATES:
        h_state = _state_rows(historical, state)
        m_state = _state_rows(oversampled, state)
        h_scramble = _scramble_rows(historical, state)
        m_scramble = _scramble_rows(oversampled, state)
        h_conv = sum(str(row["component"]) == "scramble_conversion" for row in h_scramble)
        m_conv = sum(str(row["component"]) == "scramble_conversion" for row in m_scramble)

        h_scramble_rate = len(h_scramble) / len(h_state)
        m_scramble_rate = len(m_scramble) / len(m_state)
        h_conv_per_snap = h_conv / len(h_state)
        m_conv_per_snap = m_conv / len(m_state)
        h_conv_given_scramble = h_conv / len(h_scramble) if h_scramble else 0.0
        m_conv_given_scramble = m_conv / len(m_scramble) if m_scramble else 0.0

        hs_lo, hs_hi = _wilson_interval(len(h_scramble), len(h_state))
        ms_lo, ms_hi = _wilson_interval(len(m_scramble), len(m_state))
        hc_lo, hc_hi = _wilson_interval(h_conv, len(h_scramble))
        mc_lo, mc_hi = _wilson_interval(m_conv, len(m_scramble))

        output.append(
            {
                "coarse_state": state,
                "historical_state_samples": len(h_state),
                "monster_state_samples": len(m_state),
                "historical_scramble_samples": len(h_scramble),
                "monster_scramble_samples": len(m_scramble),
                "historical_scramble_conversion_samples": h_conv,
                "monster_scramble_conversion_samples": m_conv,
                "historical_scramble_rate_per_snap": h_scramble_rate,
                "monster_scramble_rate_per_snap": m_scramble_rate,
                "monster_minus_historical_scramble_rate": m_scramble_rate - h_scramble_rate,
                "historical_conversion_rate_per_snap": h_conv_per_snap,
                "monster_conversion_rate_per_snap": m_conv_per_snap,
                "monster_minus_historical_conversion_rate_per_snap": m_conv_per_snap - h_conv_per_snap,
                "historical_conversion_given_scramble": h_conv_given_scramble,
                "monster_conversion_given_scramble": m_conv_given_scramble,
                "monster_minus_historical_conversion_given_scramble": (
                    m_conv_given_scramble - h_conv_given_scramble
                ),
                "historical_scramble_rate_wilson_low": hs_lo,
                "historical_scramble_rate_wilson_high": hs_hi,
                "monster_scramble_rate_wilson_low": ms_lo,
                "monster_scramble_rate_wilson_high": ms_hi,
                "scramble_rate_intervals_overlap": not (ms_hi < hs_lo or hs_hi < ms_lo),
                "historical_conversion_given_scramble_wilson_low": hc_lo,
                "historical_conversion_given_scramble_wilson_high": hc_hi,
                "monster_conversion_given_scramble_wilson_low": mc_lo,
                "monster_conversion_given_scramble_wilson_high": mc_hi,
                "conversion_shape_intervals_overlap": not (mc_hi < hc_lo or hc_hi < mc_lo),
                "evidence_sufficient": h_conv >= 20 and m_conv >= 50,
            }
        )
    return output


def _joint_for_state(rows: list[dict[str, object]], state: str) -> dict[str, dict[str, float]]:
    return joint_component_probabilities(
        row for row in rows if str(row["coarse_state"]) == state
    )


def _family_transition(
    baseline_joint: dict[str, dict[str, float]],
    replacement_joint: dict[str, dict[str, float]],
    *,
    mode: str,
) -> dict[str, float]:
    if mode not in {"mass_and_shape", "rate_only", "shape_only"}:
        raise ValueError(f"unsupported family transplant mode: {mode}")

    def aggregate_family(joint: dict[str, dict[str, float]]) -> dict[str, float]:
        output: dict[str, float] = defaultdict(float)
        for component in SCRAMBLE_COMPONENTS:
            for next_state, probability in joint.get(component, {}).items():
                output[next_state] += float(probability)
        return dict(output)

    baseline_family = aggregate_family(baseline_joint)
    replacement_family = aggregate_family(replacement_joint)
    baseline_mass = sum(baseline_family.values())
    replacement_mass = sum(replacement_family.values())
    target_mass = baseline_mass if mode == "shape_only" else replacement_mass

    if mode == "rate_only" or not replacement_family:
        shape_source = baseline_family
    else:
        shape_source = replacement_family
    shape_total = sum(shape_source.values())
    target_shape = (
        {key: value / shape_total for key, value in shape_source.items()}
        if shape_total > 0.0
        else {}
    )

    non_target: dict[str, float] = defaultdict(float)
    for component, distribution in baseline_joint.items():
        if component in SCRAMBLE_COMPONENTS:
            continue
        for next_state, probability in distribution.items():
            non_target[next_state] += float(probability)
    baseline_non_target_mass = sum(non_target.values())
    desired_non_target_mass = max(1.0 - target_mass, 0.0)
    scale = desired_non_target_mass / baseline_non_target_mass if baseline_non_target_mass > 1e-12 else 0.0

    output: dict[str, float] = defaultdict(float)
    for next_state, probability in non_target.items():
        output[next_state] += probability * scale
    for next_state, conditional_probability in target_shape.items():
        output[next_state] += target_mass * conditional_probability
    total = sum(output.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in output.items()}


def _stabilized_joint(
    baseline_joint: dict[str, dict[str, float]],
    oversample_joint: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    over_family_mass = sum(
        sum(oversample_joint.get(component, {}).values()) for component in SCRAMBLE_COMPONENTS
    )
    baseline_non_family_mass = sum(
        sum(distribution.values())
        for component, distribution in baseline_joint.items()
        if component not in SCRAMBLE_COMPONENTS
    )
    scale = (1.0 - over_family_mass) / baseline_non_family_mass if baseline_non_family_mass > 1e-12 else 0.0
    output: dict[str, dict[str, float]] = {}
    for component, distribution in baseline_joint.items():
        if component in SCRAMBLE_COMPONENTS:
            continue
        output[component] = {key: value * scale for key, value in distribution.items()}
    for component in SCRAMBLE_COMPONENTS:
        if component in oversample_joint:
            output[component] = dict(oversample_joint[component])
    return output


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
    hist_survival, _ = solve_series_survival(hist_transitions, start_distribution)
    mon_survival, _ = solve_series_survival(mon_transitions, start_distribution)
    gap = hist_survival - mon_survival

    evidence_map = {str(row["coarse_state"]): bool(row["evidence_sufficient"]) for row in evidence}
    hist_joint = {state: _joint_for_state(historical, state) for state in TARGET_STATES}
    base_joint = {state: _joint_for_state(baseline_monster, state) for state in TARGET_STATES}
    over_joint = {state: _joint_for_state(oversampled, state) for state in TARGET_STATES}

    output: list[dict[str, object]] = []
    for scope in (*TARGET_STATES, "d1+d2:7_10"):
        states = list(TARGET_STATES) if scope == "d1+d2:7_10" else [scope]
        if not all(evidence_map.get(state, False) for state in states):
            continue

        stabilized_transitions = {key: dict(value) for key, value in mon_transitions.items()}
        stabilized_joints: dict[str, dict[str, dict[str, float]]] = {}
        for state in states:
            stabilized_joints[state] = _stabilized_joint(base_joint[state], over_joint[state])
            stabilized_transitions[state] = _family_transition(
                base_joint[state], over_joint[state], mode="mass_and_shape"
            )
        stabilized_survival, _ = solve_series_survival(stabilized_transitions, start_distribution)

        for mode in ("mass_and_shape", "rate_only", "shape_only"):
            hybrid = {key: dict(value) for key, value in stabilized_transitions.items()}
            for state in states:
                hybrid[state] = _family_transition(
                    stabilized_joints[state], hist_joint[state], mode=mode
                )
            survival, _ = solve_series_survival(hybrid, start_distribution)
            delta = survival - stabilized_survival
            output.append(
                {
                    "scope": scope,
                    "mode": mode,
                    "states_replaced": len(states),
                    "original_monster_series_survival": mon_survival,
                    "oversample_stabilized_monster_survival": stabilized_survival,
                    "hybrid_series_survival": survival,
                    "historical_series_survival": hist_survival,
                    "absolute_recovery_vs_stabilized": delta,
                    "recovery_percentage_points": delta * 100.0,
                    "fraction_of_original_gap_recovered": delta / gap if gap > 1e-12 else 0.0,
                }
            )
    output.sort(key=lambda row: float(row["absolute_recovery_vs_stabilized"]), reverse=True)
    return output, {
        "historical_series_survival": hist_survival,
        "monster_series_survival": mon_survival,
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
    parser.add_argument("--d1-sample", type=int, default=2500)
    parser.add_argument("--d1-replicates", type=int, default=40)
    parser.add_argument("--d2-sample", type=int, default=1500)
    parser.add_argument("--d2-replicates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026091204)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    historical_raw = pl.read_parquet(args.historical).to_dicts()
    historical = _enrich(historical_raw)
    baseline_monster = _enrich(pl.read_parquet(args.baseline_monster).to_dicts())
    teams, defenses, context = _build_current_teams(args)

    oversampled_raw: list[dict[str, object]] = []
    state_settings = {
        "d1:7_10": (args.d1_sample, args.d1_replicates, args.seed + 101),
        "d2:7_10": (args.d2_sample, args.d2_replicates, args.seed + 202),
    }
    for state, (sample_count, replicates, state_seed) in state_settings.items():
        sampled = _sample_state(
            historical_raw,
            state,
            maximum=sample_count,
            seed=state_seed,
        )
        oversampled_raw.extend(
            _monster_transition_rows(
                sampled,
                teams=teams,
                defenses=defenses,
                pools=context["pools"],
                opponents=context["opponents"],
                replicates=replicates,
                seed=state_seed + 19_000_003,
            )
        )
    oversampled = _enrich(oversampled_raw)
    evidence = _evidence(historical, oversampled)
    counterfactuals, survival = _counterfactuals(
        historical, baseline_monster, oversampled, evidence
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(oversampled).write_parquet(args.out / "scramble_family_monster_rows.parquet")
    pl.DataFrame(evidence).write_csv(args.out / "scramble_family_evidence.csv")
    pl.DataFrame(counterfactuals).write_csv(args.out / "scramble_family_counterfactuals.csv")

    ledger = {
        "experiment": "MON-LEDGER-002C.4-SCRAMBLE-FAMILY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "vlourish_runtime": {
            "center": "Determine whether missing scramble conversions come from too few scrambles or from the yardage/conversion shape after a scramble occurs.",
            "boundary": "Scramble family only in d1:7_10 and d2:7_10; no production behavior changes.",
            "direction": "Separate scramble branch frequency from conversion-given-scramble before changing QB response or scramble yardage mechanics.",
            "practice": [
                "Oversample both target states until conversion counts clear the precommitted evidence threshold.",
                "Measure total scramble rate per snap and conversion probability conditional on a scramble.",
                "Counterfactually transplant scramble rate only, outcome shape only, then both.",
            ],
            "evidence": {
                **survival,
                "monster_oversample_rows": len(oversampled),
                "state_evidence": evidence,
                "counterfactuals": counterfactuals,
            },
            "reflection": "PENDING_EVIDENCE_REVIEW",
            "expansion": "Repair only the scramble submechanism demonstrated to carry the missing survival probability.",
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
        "artifact": "Monster Vlourish Scramble Family Audit",
        "experiment": ledger["experiment"],
        "target_states": list(TARGET_STATES),
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
