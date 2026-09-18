from __future__ import annotations

# Out-of-sample next-season game-level designed-rush entry rates, conditioned only
# on the player's prior-season carry count. Derived from regular-season nflverse
# weekly player stats, 2022-2025, before any 2026 Week 1 truth is consumed.
#
# Each probability is next_rush_games / next_games for players observed in
# consecutive seasons. This preserves the specialist tail instead of forcing a
# single position-wide shrinkage factor.
_ENTRY_RATES = {
    "WR": {
        "0": 82 / 2592,
        "1-3": 215 / 1768,
        "4-8": 174 / 872,
        "9+": 248 / 610,
    },
    "TE": {
        "0": 29 / 2489,
        "1-3": 37 / 553,
        "4-8": 8 / 35,
        "9+": 36 / 37,
    },
}


def carry_bin(historical_rushes: float) -> str:
    rushes = max(float(historical_rushes), 0.0)
    if rushes < 0.5:
        return "0"
    if rushes < 3.5:
        return "1-3"
    if rushes < 8.5:
        return "4-8"
    return "9+"


def empirical_gadget_entry_prior(
    position: str,
    historical_rushes: float,
) -> float:
    pos = position.upper()
    if pos not in _ENTRY_RATES:
        raise ValueError(f"Unsupported gadget-rush position: {position}")
    return float(_ENTRY_RATES[pos][carry_bin(historical_rushes)])


def entry_rate_table() -> dict[str, dict[str, float]]:
    return {
        position: {
            bucket: float(probability)
            for bucket, probability in rates.items()
        }
        for position, rates in _ENTRY_RATES.items()
    }
