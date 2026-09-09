from __future__ import annotations

from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.12)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.88)
    wr = PlayerIdentity(f"{team}-wr", f"{team} WR", "WR", usage_weight=0.55)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.45)
    return TeamIdentity(team, qb, (rb, qb), (wr, te))


def test_complete_game_event_ledger_conserves_box_score() -> None:
    for seed in range(10, 30):
        result = simulate_regulation_game(_team("away"), _team("home"), seed=seed)
        assert_event_conservation(result)


def test_anatomy_summary_is_finite_and_event_derived() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=77)
    summary = summarize_game(result)
    assert summary.snaps == len(result.plays)
    assert summary.pass_plays + summary.run_plays + summary.punts + summary.field_goal_attempts == summary.snaps
    assert summary.drives > 1
    assert summary.away_points >= 0
    assert summary.home_points >= 0
