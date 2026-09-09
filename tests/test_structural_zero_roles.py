import numpy as np

from monster.sim.allocation import _sample_role_shares
from monster.snapshot.player import PlayerState


def test_structural_zero_role_never_receives_share_when_other_role_mass_exists():
    players = (
        PlayerState("qb", "Quarterback", "QB", "A", role_uncertainty=0.20),
        PlayerState("wr1", "Receiver One", "WR", "A", role_uncertainty=0.20),
        PlayerState("wr2", "Receiver Two", "WR", "A", role_uncertainty=0.20),
    )
    shares = _sample_role_shares(
        np.random.default_rng(7),
        players,
        np.array([0.0, 0.65, 0.35]),
        worlds=5000,
    )
    assert np.all(shares[:, 0] == 0.0)
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_all_zero_role_vector_falls_back_to_uncertain_equal_prior():
    players = (
        PlayerState("a", "A", "WR", "A"),
        PlayerState("b", "B", "WR", "A"),
    )
    shares = _sample_role_shares(
        np.random.default_rng(9),
        players,
        np.array([0.0, 0.0]),
        worlds=1000,
    )
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert np.any(shares[:, 0] > 0.0)
    assert np.any(shares[:, 1] > 0.0)


def test_inactive_only_positive_role_uses_eligible_fallback_not_structural_zero():
    players = (
        PlayerState("zero", "Zero", "WR", "A", active_probability=1.0),
        PlayerState("eligible", "Eligible", "WR", "A", active_probability=0.0),
    )
    shares = _sample_role_shares(
        np.random.default_rng(11),
        players,
        np.array([0.0, 1.0]),
        worlds=100,
    )
    assert np.all(shares[:, 0] == 0.0)
    assert np.all(shares[:, 1] == 1.0)
