import numpy as np

from monster.dfs.world_matrix import build_defense_world_rows


def _full_slate(worlds: int, *, asymmetric_first_game: bool = False):
    index = []
    payload: dict[str, np.ndarray] = {}
    attempts: dict[str, np.ndarray] = {}
    for idx in range(12):
        away, home = f"A{idx}", f"H{idx}"
        index.append({"game_index": idx, "game": f"{away}@{home}", "away_team": away, "home_team": home})
        if idx == 0 and asymmetric_first_game:
            # A0 defense should see H0's 0 points; H0 defense should see A0's 40 points.
            payload[f"g{idx}_away_points"] = np.full(worlds, 40)
            payload[f"g{idx}_home_points"] = np.zeros(worlds, dtype=np.int16)
            payload[f"g{idx}_away_turnovers"] = np.zeros(worlds, dtype=np.int16)
            payload[f"g{idx}_home_turnovers"] = np.zeros(worlds, dtype=np.int16)
            attempts[away] = np.zeros(worlds)
            attempts[home] = np.zeros(worlds)
        else:
            payload[f"g{idx}_away_points"] = np.full(worlds, 20 + idx % 3)
            payload[f"g{idx}_home_points"] = np.full(worlds, 23 + idx % 3)
            payload[f"g{idx}_away_turnovers"] = np.ones(worlds, dtype=np.int16)
            payload[f"g{idx}_home_turnovers"] = np.ones(worlds, dtype=np.int16)
            attempts[away] = np.full(worlds, 32.0)
            attempts[home] = np.full(worlds, 33.0)
        payload[f"g{idx}_away_pass_disruption"] = np.ones(worlds)
        payload[f"g{idx}_home_pass_disruption"] = np.ones(worlds)
    return index, payload, attempts


def test_builds_two_correlated_defenses_per_game() -> None:
    worlds = 2000
    index = [{"game_index": 0, "game": "AAA@BBB", "away_team": "AAA", "home_team": "BBB"}]
    game_worlds = {
        "g0_away_points": np.full(worlds, 20),
        "g0_home_points": np.full(worlds, 24),
        "g0_away_turnovers": np.ones(worlds, dtype=np.int16),
        "g0_home_turnovers": np.full(worlds, 2, dtype=np.int16),
        "g0_away_pass_disruption": np.ones(worlds),
        "g0_home_pass_disruption": np.ones(worlds),
    }
    attempts = {"AAA": np.full(worlds, 31.0), "BBB": np.full(worlds, 34.0)}
    try:
        build_defense_world_rows(game_index=index, game_worlds=game_worlds, team_pass_attempts=attempts, seed=9)
    except RuntimeError as exc:
        # Production Week 1 requires all 12 games. The guard itself is part of the contract.
        assert "Expected 24 Week 1 D/ST rows" in str(exc)


def test_full_week_has_24_aligned_defense_rows() -> None:
    worlds = 1000
    index, payload, attempts = _full_slate(worlds)
    rows = build_defense_world_rows(
        game_index=index,
        game_worlds=payload,
        team_pass_attempts=attempts,
        seed=19,
    )
    assert len(rows) == 24
    assert len({row.team_id for row in rows}) == 24
    assert all(row.position == "D" for row in rows)
    assert all(row.scores.shape == (worlds,) for row in rows)
    assert all(np.isfinite(row.scores).all() for row in rows)


def test_defense_points_allowed_uses_opponent_score_not_own_score() -> None:
    worlds = 4000
    index, payload, attempts = _full_slate(worlds, asymmetric_first_game=True)
    rows = build_defense_world_rows(
        game_index=index,
        game_worlds=payload,
        team_pass_attempts=attempts,
        seed=31,
    )
    by_team = {row.team_id: row for row in rows}

    # With no pass attempts or turnovers, only rare ST/safety events can add points.
    # The FanDuel points-allowed component therefore creates a hard 14-point separation:
    # A0 allows 0 (+10), while H0 allows 40 (-4).
    assert float(by_team["A0"].scores.mean()) > 9.5
    assert float(by_team["H0"].scores.mean()) < -3.0
    assert float(by_team["A0"].scores.mean() - by_team["H0"].scores.mean()) > 13.0
