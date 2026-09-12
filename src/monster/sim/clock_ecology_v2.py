from __future__ import annotations

import numpy as np


def sample_snap_cadence_v2(*, hurry: float, rng: np.random.Generator) -> int:
    """Sample game-clock runoff to the next snap for Reality Loop v2.

    The original v1.3 center of 39 seconds produced only ~112 scrimmage plays/game even after
    drive conversion/survival matched NFL reality closely. That indicates a clock-volume error,
    not a need to manufacture more successful offense. This cadence recenters ordinary NFL
    between-snap runoff while preserving hurry-up acceleration and broad stochastic variation.

    Stopped-clock plays continue to be handled by ``_event_elapsed_seconds``; this function is
    only the underlying live-clock cadence prior.
    """

    hurry_level = float(np.clip(hurry, 0.0, 1.0))
    center = 35.5 - 20.0 * hurry_level
    spread = 10.5 - 2.0 * hurry_level
    return int(np.clip(rng.normal(center, spread), 6.0, 50.0))
