from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence


ABSORB_CONVERTED = "ABSORB_CONVERTED"
ABSORB_TOUCHDOWN = "ABSORB_TOUCHDOWN"
ABSORB_FAILED = "ABSORB_FAILED"
ABSORBING_STATES = {ABSORB_CONVERTED, ABSORB_TOUCHDOWN, ABSORB_FAILED}


def distance_bucket(distance: float) -> str:
    value = max(float(distance), 0.0)
    if value <= 1.0:
        return "1"
    if value <= 3.0:
        return "2_3"
    if value <= 6.0:
        return "4_6"
    if value <= 10.0:
        return "7_10"
    return "11_plus"


def field_zone(yardline_from_own: float) -> str:
    value = float(yardline_from_own)
    if value < 20.0:
        return "own_1_19"
    if value < 40.0:
        return "own_20_39"
    if value < 55.0:
        return "own_40_to_plus_46"
    if value < 70.0:
        return "plus_45_to_31"
    if value < 80.0:
        return "plus_30_to_21"
    return "red_zone"


def coarse_state_key(down: int, distance: float) -> str:
    return f"d{int(down)}:{distance_bucket(distance)}"


def fine_state_key(down: int, distance: float, yardline_from_own: float) -> str:
    return f"{coarse_state_key(down, distance)}:{field_zone(yardline_from_own)}"


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def normalized(counter: Mapping[str, int | float]) -> dict[str, float]:
    total = sum(float(value) for value in counter.values())
    if total <= 0.0:
        return {}
    return {key: float(value) / total for key, value in counter.items()}


def transition_probabilities(
    rows: Iterable[Mapping[str, object]],
    *,
    state_field: str = "coarse_state",
    next_field: str = "next_state",
) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        state = str(row[state_field])
        next_state = str(row[next_field])
        counts[state][next_state] += 1
    return {state: normalized(counter) for state, counter in counts.items()}


def solve_series_survival(
    transitions: Mapping[str, Mapping[str, float]],
    start_distribution: Mapping[str, float],
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> tuple[float, dict[str, float]]:
    """Solve probability of earning a new series or scoring before series failure.

    ``ABSORB_CONVERTED`` and ``ABSORB_TOUCHDOWN`` are successful absorbing outcomes.
    ``ABSORB_FAILED`` is unsuccessful. Penalty replay loops and other same-state transitions
    are handled through fixed-point iteration rather than special cases.
    """

    states = set(transitions) | set(start_distribution)
    values = {state: 0.0 for state in states}
    values[ABSORB_CONVERTED] = 1.0
    values[ABSORB_TOUCHDOWN] = 1.0
    values[ABSORB_FAILED] = 0.0

    for _ in range(max_iterations):
        delta = 0.0
        updated = dict(values)
        for state in states:
            distribution = transitions.get(state)
            if not distribution:
                new_value = 0.0
            else:
                new_value = 0.0
                for next_state, probability in distribution.items():
                    if next_state == ABSORB_CONVERTED or next_state == ABSORB_TOUCHDOWN:
                        next_value = 1.0
                    elif next_state == ABSORB_FAILED:
                        next_value = 0.0
                    else:
                        next_value = values.get(next_state, 0.0)
                    new_value += float(probability) * next_value
            delta = max(delta, abs(new_value - values.get(state, 0.0)))
            updated[state] = new_value
        values = updated
        if delta <= tolerance:
            break

    total_weight = sum(float(value) for value in start_distribution.values())
    if total_weight <= 0.0:
        return 0.0, values
    survival = sum(
        float(weight) * values.get(state, 0.0) for state, weight in start_distribution.items()
    ) / total_weight
    return survival, values


def replace_transition_family(
    baseline: Mapping[str, Mapping[str, float]],
    replacement: Mapping[str, Mapping[str, float]],
    states: Sequence[str],
) -> dict[str, dict[str, float]]:
    hybrid = {state: dict(distribution) for state, distribution in baseline.items()}
    for state in states:
        if state in replacement:
            hybrid[state] = dict(replacement[state])
    return hybrid


def joint_component_probabilities(
    rows: Iterable[Mapping[str, object]],
    *,
    component_field: str = "component",
    next_field: str = "next_state",
) -> dict[str, dict[str, float]]:
    """Return P(component, next_state) for one starting-state population."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    for row in rows:
        component = str(row[component_field])
        next_state = str(row[next_field])
        counts[component][next_state] += 1
        total += 1
    if total <= 0:
        return {}
    return {
        component: {next_state: count / total for next_state, count in counter.items()}
        for component, counter in counts.items()
    }


def aggregate_joint_transition(
    joint: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    output: dict[str, float] = defaultdict(float)
    for distribution in joint.values():
        for next_state, probability in distribution.items():
            output[next_state] += float(probability)
    return dict(output)


def transplant_component_transition(
    baseline_joint: Mapping[str, Mapping[str, float]],
    replacement_joint: Mapping[str, Mapping[str, float]],
    component: str,
    *,
    mode: str = "mass_and_shape",
) -> dict[str, float]:
    """Transplant one mechanism component into a starting-state transition distribution.

    ``mass_and_shape`` replaces both the component probability and its conditional next-state
    distribution with historical evidence. ``rate_only`` replaces component probability while
    retaining Monster's conditional next-state shape. ``shape_only`` retains Monster's component
    probability but replaces its conditional next-state shape. Non-target Monster outcomes are
    proportionally rescaled only when component mass changes, so their relative structure remains
    untouched.
    """

    if mode not in {"mass_and_shape", "rate_only", "shape_only"}:
        raise ValueError(f"unsupported component transplant mode: {mode}")

    baseline_target = dict(baseline_joint.get(component, {}))
    replacement_target = dict(replacement_joint.get(component, {}))
    baseline_mass = sum(float(value) for value in baseline_target.values())
    replacement_mass = sum(float(value) for value in replacement_target.values())

    if mode == "shape_only":
        target_mass = baseline_mass
    else:
        target_mass = replacement_mass

    if mode == "rate_only" or not replacement_target:
        target_shape = normalized(baseline_target)
    else:
        target_shape = normalized(replacement_target)

    non_target_joint = {
        name: dict(distribution)
        for name, distribution in baseline_joint.items()
        if name != component
    }
    baseline_non_target_mass = 1.0 - baseline_mass
    desired_non_target_mass = max(1.0 - target_mass, 0.0)
    if baseline_non_target_mass > 1e-12:
        non_target_scale = desired_non_target_mass / baseline_non_target_mass
    else:
        non_target_scale = 0.0

    output: dict[str, float] = defaultdict(float)
    for distribution in non_target_joint.values():
        for next_state, probability in distribution.items():
            output[next_state] += float(probability) * non_target_scale
    for next_state, conditional_probability in target_shape.items():
        output[next_state] += target_mass * float(conditional_probability)

    total = sum(output.values())
    if total <= 0.0:
        return {}
    return {next_state: probability / total for next_state, probability in output.items()}
