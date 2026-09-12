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
