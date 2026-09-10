from __future__ import annotations

from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.12)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.75)
    wr1 = PlayerIdentity(f"{team}-wr1", f"{team} WR1", "WR", usage_weight=0.45)
    wr2 = PlayerIdentity(f"{team}-wr2", f"{team} WR2", "WR", usage_weight=0.30)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.25)
    return TeamIdentity(team, qb, (rb, qb), (wr1, wr2, te))


def test_drive_survival_trace_is_observational_and_conserved() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=2026091031)
    assert result.drive_traces

    for trace in result.drive_traces:
        down_snaps = (
            trace.first_down_snaps
            + trace.second_down_snaps
            + trace.third_down_snaps
            + trace.fourth_down_snaps
        )
        assert down_snaps == trace.scrimmage_plays
        assert trace.series_started == trace.first_down_snaps
        assert 0 <= trace.series_converted <= trace.first_downs
        assert 0 <= trace.third_down_conversions <= trace.third_down_snaps
        assert 0 <= trace.third_and_long_snaps <= trace.third_down_snaps
        assert 0 <= trace.third_and_long_conversions <= trace.third_and_long_snaps
        assert 0 <= trace.early_down_5plus_gains <= (
            trace.first_down_snaps + trace.second_down_snaps
        )
        assert trace.third_down_distance_total >= 0.0


def test_drive_survival_trace_does_not_change_seeded_game_world() -> None:
    first = simulate_regulation_game(_team("away"), _team("home"), seed=2026091032)
    second = simulate_regulation_game(_team("away"), _team("home"), seed=2026091032)
    assert first.final_state == second.final_state
    assert first.plays == second.plays
    assert first.drive_traces == second.drive_traces
