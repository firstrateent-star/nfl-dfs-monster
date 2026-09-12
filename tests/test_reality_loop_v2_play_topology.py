from __future__ import annotations

import numpy as np

from monster.sim.play_anatomy import resolve_run_contact


def test_open_field_contact_preserves_scramble_sized_positive_tail() -> None:
    """Generic open-field contact must not collapse every QB escape into 3-6 yards.

    The live play kernel routes scrambles through this contact resolver. This gate is broad on
    purpose: nflverse calibration belongs in the historical audit, while the unit test protects
    the causal topology that successful escapes can produce 5+, 10+ and 15+ gains.
    """

    rng = np.random.default_rng(20260912)
    yards = np.asarray(
        [
            resolve_run_contact(
                penetration_probability=0.08,
                runner_power=1.0,
                tackling=1.0,
                explosiveness=1.0,
                rng=rng,
            ).total_yards
            for _ in range(20_000)
        ],
        dtype=float,
    )

    assert float(np.mean(yards >= 5.0)) > 0.48
    assert float(np.mean(yards >= 10.0)) > 0.16
    assert float(np.mean(yards >= 15.0)) > 0.065
    assert float(np.mean(yards < 0.0)) > 0.01


def test_explosiveness_moves_open_field_tail_without_removing_failure() -> None:
    low_rng = np.random.default_rng(771)
    high_rng = np.random.default_rng(771)
    low = np.asarray(
        [
            resolve_run_contact(
                penetration_probability=0.08,
                runner_power=0.9,
                tackling=1.05,
                explosiveness=0.75,
                rng=low_rng,
            ).total_yards
            for _ in range(12_000)
        ]
    )
    high = np.asarray(
        [
            resolve_run_contact(
                penetration_probability=0.08,
                runner_power=1.1,
                tackling=0.95,
                explosiveness=1.25,
                rng=high_rng,
            ).total_yards
            for _ in range(12_000)
        ]
    )

    assert float(np.mean(high >= 10.0)) > float(np.mean(low >= 10.0))
    assert float(np.mean(high >= 15.0)) > float(np.mean(low >= 15.0))
    assert float(np.mean(low < 0.0)) > 0.0
    assert float(np.mean(high < 0.0)) > 0.0
