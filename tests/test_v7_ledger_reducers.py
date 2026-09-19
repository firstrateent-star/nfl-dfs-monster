from __future__ import annotations

from monster.reality.engine import RealityGameRequest, V6CompatibilityEngine
from monster.reality.reducers import final_score, reduce_player_box_scores
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.15)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.85)
    wr = PlayerIdentity(f"{team}-wr", f"{team} WR", "WR", usage_weight=0.60)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.40)
    return TeamIdentity(team, qb, (rb, qb), (wr, te))


def test_v7_ledger_reduces_exact_v6_offensive_box_scores() -> None:
    result = V6CompatibilityEngine().run(
        RealityGameRequest.compatibility(
            away=_team("AWY"),
            home=_team("HME"),
            seed=7005001,
        )
    )
    reduced = reduce_player_box_scores(result.ledger)
    assert reduced == result.football.player_stats


def test_v7_ledger_final_score_matches_engine_final_state() -> None:
    result = V6CompatibilityEngine().run(
        RealityGameRequest.compatibility(
            away=_team("AWY"),
            home=_team("HME"),
            seed=7005002,
        )
    )
    assert final_score(result.ledger) == (
        result.football.final_state.away_score,
        result.football.final_state.home_score,
    )
