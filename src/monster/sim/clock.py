from __future__ import annotations

from dataclasses import replace

from monster.sim.football_state import FootballState

REGULATION_SECONDS = 3600
QUARTER_SECONDS = 900


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
    """Consume finite regulation time and update quarter deterministically.

    The game owns one clock. Drives and plays consume it; they cannot independently create
    possession volume. Overtime is deliberately excluded from this first regulation kernel.
    """
    elapsed = max(int(elapsed_seconds), 0)
    remaining = max(state.seconds_remaining - elapsed, 0)
    return replace(state, seconds_remaining=remaining, quarter=quarter_from_clock(remaining))


def regulation_complete(state: FootballState) -> bool:
    return state.seconds_remaining <= 0


def halftime_crossed(before: FootballState, after: FootballState) -> bool:
    return before.seconds_remaining > 1800 >= after.seconds_remaining


def two_minute_warning_crossed(before: FootballState, after: FootballState) -> bool:
    boundaries = (1920, 120)
    return any(before.seconds_remaining > boundary >= after.seconds_remaining for boundary in boundaries)
