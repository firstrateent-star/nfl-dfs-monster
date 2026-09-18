from __future__ import annotations

import numpy as np

# Market-blind empirical conditional-carry distributions from regular-season
# nflverse weekly player stats, 2022-2025. These are conditional on a WR/TE
# recording at least one carry in a game. Source audit:
# scripts/audit_gadget_rush_incidence_oos_v637.py
#
# WR: 1,291 entry games; mean 1.36096 carries; median 1; p90 2.
# TE:   175 entry games; mean 2.32 carries; median 1; p90 6.
_WR_COUNTS = {
    1: 972,
    2: 221,
    3: 68,
    4: 18,
    5: 9,
    6: 1,
    8: 2,
}
_TE_COUNTS = {
    1: 117,
    2: 16,
    3: 9,
    4: 7,
    5: 7,
    6: 4,
    7: 4,
    9: 5,
    10: 2,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
}


def _distribution(
    counts: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sorted(counts), dtype=int)
    weights = np.asarray([counts[int(value)] for value in values], dtype=float)
    return values, weights / weights.sum()


WR_CARRY_VALUES, WR_CARRY_PROBABILITIES = _distribution(_WR_COUNTS)
TE_CARRY_VALUES, TE_CARRY_PROBABILITIES = _distribution(_TE_COUNTS)


def conditional_carry_mean(position: str) -> float:
    pos = position.upper()
    if pos == "WR":
        return float(np.dot(WR_CARRY_VALUES, WR_CARRY_PROBABILITIES))
    if pos == "TE":
        return float(np.dot(TE_CARRY_VALUES, TE_CARRY_PROBABILITIES))
    raise ValueError(f"Unsupported gadget-rush position: {position}")


def sample_conditional_gadget_carries(
    position: str,
    *,
    rng: np.random.Generator,
) -> int:
    pos = position.upper()
    if pos == "WR":
        return int(rng.choice(WR_CARRY_VALUES, p=WR_CARRY_PROBABILITIES))
    if pos == "TE":
        return int(rng.choice(TE_CARRY_VALUES, p=TE_CARRY_PROBABILITIES))
    raise ValueError(f"Unsupported gadget-rush position: {position}")
