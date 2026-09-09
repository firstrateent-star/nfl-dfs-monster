from __future__ import annotations

from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.10, explosive=1.15)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.90)
    wr = PlayerIdentity(f"{team}-wr", f"{team} WR", "WR", usage_weight=0.65)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.35)
    return TeamIdentity(team, qb, (rb, qb), (wr, te), pass_protection=0.88)


def _defense(team: str) -> DefensiveUnit:
    edge = DefensiveIdentity(f"{team}-edge", "Edge", "EDGE", pass_rush=1.25, run_defense=1.15, tackling=1.1)
    cb = DefensiveIdentity(f"{team}-cb", "Corner", "CB", coverage=1.1, tackling=1.0, ball_hawk=1.1)
    return DefensiveUnit(front=(edge,), coverage=(cb,), pressure_rate=0.10)


def test_scrambles_are_rushes_not_pass_attempts() -> None:
    found = False
    for seed in range(1, 80):
        result = simulate_regulation_game(_team("away"), _team("home"), away_defense=_defense("away"), home_defense=_defense("home"), seed=seed)
        scrambles = [play for play in result.plays if play.pass_result == PassResult.SCRAMBLE]
        if not scrambles:
            continue
        found = True
        for play in scrambles:
            qb = result.player_stats[play.rusher_id]
            assert play.rusher_id == play.passer_id
            assert play.run_lane == "qb"
            assert qb.rush_attempts >= 1
        break
    assert found


def test_rich_events_expose_catchpoint_and_contact_anatomy() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), away_defense=_defense("away"), home_defense=_defense("home"), seed=14)
    assert any(play.catchpoint_result is not None for play in result.plays)
    assert any(play.contact_result is not None for play in result.plays)
    assert any(play.air_yards != 0.0 for play in result.plays if play.target_id is not None)


def test_defensive_box_score_is_owned_by_event_defenders() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), away_defense=_defense("away"), home_defense=_defense("home"), seed=18)
    assert result.defensive_stats is not None
    event_defenders = {play.primary_defender_id for play in result.plays if play.primary_defender_id}
    assert set(result.defensive_stats).issubset(event_defenders)
    assert sum(box.pressures for box in result.defensive_stats.values()) == sum(play.pressured for play in result.plays if play.primary_defender_id)
