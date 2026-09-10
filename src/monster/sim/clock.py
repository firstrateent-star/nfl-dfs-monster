from __future__ import annotations

from dataclasses import replace

from monster.sim.football_state import FootballState

REGULATION_SECONDS = 3600
QUARTER_SECONDS = 900
OVERTIME_SECONDS = 600


def quarter_from_clock(seconds_remaining: int) -> int:
    """Return regulation quarter from one monotonic 3600-second game clock."""
    remaining = max(min(int(seconds_remaining), REGULATION_SECONDS), 0)
    if remaining > 2700:
        return 1
    if remaining > 1800:
        return 2
    if remaining > 900:
        return 3
    return 4


def seconds_remaining_in_quarter(seconds_remaining: int) -> int:
    remaining = max(min(int(seconds_remaining), REGULATION_SECONDS), 0)
    if remaining == 0:
        return 0
    return ((remaining - 1) % QUARTER_SECONDS) + 1


def advance_game_clock(state: FootballState, elapsed_seconds: int) -> FootballState:
    """Consume finite game time and update the active period deterministically.

    Regulation owns one monotonic 3600-second clock. A regular-season overtime world owns
    a separate 600-second clock while retaining quarter=5 so downstream decision policy can
    distinguish overtime from regulation.
    """
    elapsed = max(int(elapsed_seconds), 0)
    remaining = max(state.seconds_remaining - elapsed, 0)
    if state.quarter == 5:
        return replace(state, seconds_remaining=remaining, quarter=5)
    return replace(state, seconds_remaining=remaining, quarter=quarter_from_clock(remaining))


def regulation_complete(state: FootballState) -> bool:
    return state.quarter <= 4 and state.seconds_remaining <= 0


def overtime_complete(state: FootballState) -> bool:
    return state.quarter == 5 and state.seconds_remaining <= 0


def halftime_crossed(before: FootballState, after: FootballState) -> bool:
    return before.quarter <= 2 and before.seconds_remaining > 1800 >= after.seconds_remaining


def two_minute_warning_crossed(before: FootballState, after: FootballState) -> bool:
    if before.quarter == 5:
        return before.seconds_remaining > 120 >= after.seconds_remaining
    boundaries = (1920, 120)
    return any(before.seconds_remaining > boundary >= after.seconds_remaining for boundary in boundaries)
