from __future__ import annotations

import numpy as np

from monster.sim.clock_ecology_v2 import sample_snap_cadence_v2


def _mean(hurry: float, seed: int = 20260912) -> float:
    rng = np.random.default_rng(seed)
    values = [sample_snap_cadence_v2(hurry=hurry, rng=rng) for _ in range(20_000)]
    return float(np.mean(values))


def test_normal_cadence_is_faster_than_legacy_39_second_center() -> None:
    mean = _mean(0.0)
    assert 34.0 < mean < 36.5


def test_hurry_up_materially_accelerates_snap_cadence() -> None:
    normal = _mean(0.0)
    hurry = _mean(1.0)
    assert hurry < 18.0
    assert normal - hurry > 17.0


def test_cadence_remains_bounded() -> None:
    rng = np.random.default_rng(7)
    values = [sample_snap_cadence_v2(hurry=0.5, rng=rng) for _ in range(20_000)]
    assert min(values) >= 6
    assert max(values) <= 50
