from __future__ import annotations

from monster.reality.ledger import RealityLedger
from monster.sim.game_loop_v13 import PlayerBoxScore
from monster.sim.play_kernel import PassResult, PlayType


def reduce_player_box_scores(ledger: RealityLedger) -> dict[str, PlayerBoxScore]:
    """Reduce offensive player box scores from the immutable snap ledger.

    This intentionally mirrors the football scoring semantics, not DFS scoring.
    The reducer gives v7 a testable path toward making the event ledger the
    authoritative source of box scores rather than a parallel telemetry stream.
    """

    stats: dict[str, PlayerBoxScore] = {}

    def box(player_id: str | None) -> PlayerBoxScore | None:
        if player_id is None:
            return None
        return stats.setdefault(player_id, PlayerBoxScore())

    for record in ledger.snap_records:
        play = record.play
        if play is None:
            continue
        passer = box(play.passer_id)
        target = box(play.target_id)
        rusher = box(play.rusher_id)
        fumbler = box(play.fumbler_id)

        is_throw = (
            play.play_type == PlayType.PASS.value
            and play.pass_result not in {
                PassResult.SACK.value,
                PassResult.SCRAMBLE.value,
            }
        )
        if is_throw and passer is not None:
            passer.pass_attempts += 1
            if play.pass_result == PassResult.COMPLETE.value:
                passer.completions += 1
                passer.passing_yards += play.yards
                passer.passing_tds += int(play.touchdown)
            elif play.pass_result == PassResult.INTERCEPTION.value:
                passer.interceptions += 1

        if target is not None:
            target.targets += 1
            if play.pass_result == PassResult.COMPLETE.value:
                target.receptions += 1
                target.receiving_yards += play.yards
                target.receiving_tds += int(play.touchdown)

        if rusher is not None:
            rusher.rush_attempts += 1
            rusher.rushing_yards += play.yards
            rusher.rushing_tds += int(play.touchdown)

        if fumbler is not None:
            fumbler.fumbles_lost += 1

    return stats


def final_score(ledger: RealityLedger) -> tuple[int, int]:
    """Return away/home score from the ledger's terminal football state."""

    ledger.assert_contract()
    final = ledger.records[-1].post_state
    if final is None or final.away_score is None or final.home_score is None:
        raise ValueError("game-final ledger record must contain a complete scoreboard")
    return int(final.away_score), int(final.home_score)
