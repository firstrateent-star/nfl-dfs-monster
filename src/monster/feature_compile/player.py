from __future__ import annotations


def bmi(height_in: float | None, weight_lbs: float | None) -> float | None:
    if not height_in or not weight_lbs or height_in <= 0:
        return None
    return 703.0 * weight_lbs / (height_in * height_in)


def speed_score(weight_lbs: float | None, forty_time: float | None) -> float | None:
    if not weight_lbs or not forty_time or forty_time <= 0:
        return None
    return (weight_lbs * 200.0) / (forty_time ** 4)


def age_on_date(birth_date, game_date) -> float | None:
    if birth_date is None:
        return None
    return (game_date - birth_date).days / 365.2425
