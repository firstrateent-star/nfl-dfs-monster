from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from monster.reality import opportunity
from monster.reality.opportunity import (
    choose_rusher_for_geometry_v7,
    eligible_rushers_for_geometry_v7,
    participation_assignment_weights,
)
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _player(player_id: str, position: str, usage: float) -> PlayerIdentity:
    return PlayerIdentity(player_id, player_id, position, usage_weight=usage)


def test_participation_fallback_is_invariant_to_preallocated_usage_share() -> None:
    players = (_player("high", "WR", 0.99), _player("low", "WR", 0.01))
    weights = participation_assignment_weights(
        players,
        category="short_0_5",
        attempts={},
        shrinkage_samples=45.0,
    )
    assert weights.tolist() == pytest.approx([0.5, 0.5])


def test_concept_history_can_tilt_assignment_only_among_live_participants() -> None:
    players = (
        _player("a", "WR", 0.05),
        _player("b", "WR", 0.90),
        _player("c", "TE", 0.05),
    )
    weights = participation_assignment_weights(
        players,
        category="deep_20_39",
        attempts={
            ("a", "deep_20_39"): 30,
            ("c", "deep_20_39"): 2,
            ("not-on-snap", "deep_20_39"): 1000,
        },
        shrinkage_samples=20.0,
    )
    assert weights[0] > weights[2] > weights[1]
    assert float(weights.sum()) == pytest.approx(1.0)


def test_non_sneak_qb_requires_geometry_evidence_to_enter_designed_run_tree() -> None:
    rb = _player("rb", "RB", 0.50)
    qb = _player("qb", "QB", 0.50)
    ecology = SimpleNamespace(rusher_geometry_attempts={})
    assert eligible_rushers_for_geometry_v7((rb, qb), ecology, "left_edge") == (rb,)

    ecology = SimpleNamespace(rusher_geometry_attempts={("qb", "left_edge"): 5})
    eligible = eligible_rushers_for_geometry_v7((rb, qb), ecology, "left_edge")
    assert eligible == (rb, qb)


def test_qb_sneak_is_structurally_owned_by_qb() -> None:
    rb = _player("rb", "RB", 0.99)
    qb = _player("qb", "QB", 0.01)
    ecology = SimpleNamespace(rusher_geometry_attempts={})
    selected = choose_rusher_for_geometry_v7(
        (rb, qb),
        ecology,
        "qb_sneak",
        np.random.default_rng(1),
    )
    assert selected.player_id == "qb"


def test_live_field_read_does_not_reintroduce_usage_share(monkeypatch) -> None:
    high = _player("high", "WR", 0.99)
    low = _player("low", "WR", 0.01)
    qb = _player("qb", "QB", 1.0)
    offense = TeamIdentity(
        "OFF",
        qb,
        (_player("rb", "RB", 1.0), qb),
        (high, low),
    )

    def fake_matchup(receiver, defense, **kwargs):
        return SimpleNamespace(
            completion_probability=0.65,
            yards_multiplier=1.0,
            qb_read_quality=1.0,
        )

    monkeypatch.setattr(opportunity, "resolve_pass_matchup", fake_matchup)
    rng = np.random.default_rng(7003001)
    counts = {"high": 0, "low": 0}
    for _ in range(1000):
        target, _ = opportunity.field_read_target_v7(
            offense,
            SimpleNamespace(),
            rng,
            preferred=None,
            fatigue={},
            responsibility_key="test",
        )
        counts[target.player_id] += 1

    high_rate = counts["high"] / 1000
    assert 0.45 <= high_rate <= 0.55
