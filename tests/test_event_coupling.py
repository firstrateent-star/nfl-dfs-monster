from __future__ import annotations

import numpy as np

from monster.sim.event_coupling import _capacity_weighted_allocation


def test_capacity_weighted_allocation_conserves_and_respects_source_events() -> None:
    rng = np.random.default_rng(17)
    totals = np.array([2, 1, 0, 3], dtype=int)
    capacities = np.array(
        [[2, 1, 0], [0, 2, 1], [0, 0, 0], [1, 1, 2]], dtype=int
    )
    weights = capacities.astype(float) * np.array([3.0, 1.0, 0.5])[None, :]
    out = _capacity_weighted_allocation(rng, totals, capacities, weights)
    assert np.array_equal(out.sum(axis=1), totals)
    assert np.all(out <= capacities)


def test_capacity_weighted_allocation_rejects_impossible_football_state() -> None:
    rng = np.random.default_rng(19)
    totals = np.array([2], dtype=int)
    capacities = np.array([[1, 0]], dtype=int)
    weights = np.array([[1.0, 0.0]])
    try:
        _capacity_weighted_allocation(rng, totals, capacities, weights)
    except ValueError as exc:
        assert "exceeds eligible source events" in str(exc)
    else:
        raise AssertionError("impossible touchdown state must fail closed")
