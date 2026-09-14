from __future__ import annotations

from datetime import date

import run_reality_loop_v63 as v63


MATCHUP = (("DEN", "KC"),)
GAME_DATE = date(2026, 9, 14)


def main() -> None:
    integrated = v63.v61.runner.integrated
    original_matchups = integrated.MATCHUPS
    original_game_date = integrated.GAME_DATE
    try:
        integrated.MATCHUPS = MATCHUP
        integrated.GAME_DATE = GAME_DATE
        v63.main()
    finally:
        integrated.MATCHUPS = original_matchups
        integrated.GAME_DATE = original_game_date


if __name__ == "__main__":
    main()
