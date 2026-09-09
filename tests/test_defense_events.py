import numpy as np
import pytest

from monster.dfs.defense_events import simulate_defense_events


def test_defense_events_are_reproducible_and_shape_aligned() -> None:
    attempts = np.full(20000, 33.0)
    turnovers = np.ones(20000, dtype=np.int16)
    disruption = np.ones(20000)
    first = simulate_defense_events(
        opponent_pass_attempts=attempts,
        opponent_turnovers=turnovers,
        pass_disruption=disruption,
        seed=7,
    )
    second = simulate_defense_events(
        opponent_pass_attempts=attempts,
        opponent_turnovers=turnovers,
        pass_disruption=disruption,
        seed=7,
    )
    for key in ("sacks", "defensive_touchdowns", "special_teams_touchdowns", "safeties"):
        a = getattr(first, key)
        b = getattr(second, key)
        assert a.shape == attempts.shape
        np.testing.assert_array_equal(a, b)


def test_defensive_td_requires_a_takeaway() -> None:
    worlds = 50000
    out = simulate_defense_events(
        opponent_pass_attempts=np.full(worlds, 33.0),
        opponent_turnovers=np.zeros(worlds, dtype=np.int16),
        pass_disruption=np.ones(worlds),
        seed=11,
    )
    assert int(out.defensive_touchdowns.sum()) == 0


def test_sack_anchor_and_disruption_direction() -> None:
    worlds = 100000
    attempts = np.full(worlds, 33.0)
    turnovers = np.ones(worlds, dtype=np.int16)
    low = simulate_defense_events(
        opponent_pass_attempts=attempts,
        opponent_turnovers=turnovers,
        pass_disruption=np.full(worlds, 0.80),
        seed=13,
    )
    neutral = simulate_defense_events(
        opponent_pass_attempts=attempts,
        opponent_turnovers=turnovers,
        pass_disruption=np.ones(worlds),
        seed=13,
    )
    high = simulate_defense_events(
        opponent_pass_attempts=attempts,
        opponent_turnovers=turnovers,
        pass_disruption=np.full(worlds, 1.20),
        seed=13,
    )
    assert 2.15 <= float(neutral.sacks.mean()) <= 2.45
    assert float(low.sacks.mean()) < float(neutral.sacks.mean()) < float(high.sacks.mean())


def test_rare_event_anchors_are_reasonable() -> None:
    worlds = 200000
    out = simulate_defense_events(
        opponent_pass_attempts=np.full(worlds, 33.0),
        opponent_turnovers=np.ones(worlds, dtype=np.int16),
        pass_disruption=np.ones(worlds),
        seed=17,
    )
    assert 0.065 <= float(out.defensive_touchdowns.mean()) <= 0.095
    assert 0.018 <= float(out.special_teams_touchdowns.mean()) <= 0.032
    assert 0.018 <= float(out.safeties.mean()) <= 0.032


def test_misaligned_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="correlated world shape"):
        simulate_defense_events(
            opponent_pass_attempts=np.ones(2),
            opponent_turnovers=np.ones(3, dtype=np.int16),
            pass_disruption=np.ones(2),
            seed=1,
        )
