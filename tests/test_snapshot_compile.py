from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    TeamMechanismInputs,
)
from monster.snapshot.compile import compile_game_snapshot
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool(team_id: str, opponent_id: str) -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id=team_id,
        players=(
            PlayerState(
                player_id=f"{team_id}-wr",
                display_name=f"{team_id} WR",
                position="WR",
                team_id=team_id,
                target_share=0.28,
                catch_rate=0.66,
            ),
            PlayerState(
                player_id=f"{team_id}-rb",
                display_name=f"{team_id} RB",
                position="RB",
                team_id=team_id,
                rush_share=0.62,
            ),
        ),
    )


def test_snapshot_compiler_applies_player_and_team_mechanisms():
    game = GameState(
        "A@H",
        TeamState("A", "H"),
        TeamState("H", "A"),
        dome=False,
    )
    snapshot = compile_game_snapshot(
        game,
        _pool("A", "H"),
        _pool("H", "A"),
        away_team_inputs=TeamMechanismInputs(
            team_madden_ovr=90.0,
            offensive_line_index=2.0,
            opponent_front_index=-1.0,
            wind_mph=20.0,
        ),
        player_inputs={
            "A-wr": PlayerMechanismInputs(
                forty_time=4.30,
                madden_speed=97.0,
                madden_catching=92.0,
            )
        },
    )
    away_wr = snapshot.away_pool.players[0]
    assert away_wr.explosive_modifier > 1.0
    assert snapshot.game.away.physical_madden_effect > 0.0
    assert snapshot.game.away.weather_effect < 0.0
    assert "A-wr" in snapshot.player_traces
    assert "A" in snapshot.team_traces


def test_unprovided_features_remain_neutral_in_snapshot():
    game = GameState("A@H", TeamState("A", "H"), TeamState("H", "A"))
    away_pool = _pool("A", "H")
    home_pool = _pool("H", "A")
    snapshot = compile_game_snapshot(game, away_pool, home_pool)
    assert snapshot.game == game
    assert snapshot.away_pool == away_pool
    assert snapshot.home_pool == home_pool
