from __future__ import annotations

import numpy as np

from monster.sim.return_channels_v632 import (
    ChannelReturnPriorsV632,
    ReturnDistanceProfileV632,
    channel_priors_from_policy_row,
    simulate_kickoff_v632,
    simulate_punt_v632,
)


def _profile(mean: float = 12.0, zero: float = 0.0) -> ReturnDistanceProfileV632:
    return ReturnDistanceProfileV632(
        zero_rate=zero,
        mean=mean,
        sd=2.0,
        p20=0.0,
        p40=0.0,
        p60=0.0,
        p80=0.0,
    )


def _priors(*, punt_live: float, kickoff_touchback: float, kickoff_live: float) -> ChannelReturnPriorsV632:
    return ChannelReturnPriorsV632(
        fumble_recovery=_profile(),
        punt_touchback_rate=0.0,
        punt_live_return_rate=punt_live,
        punt_live_return=_profile(mean=14.0),
        kickoff_touchback_rate=kickoff_touchback,
        kickoff_live_return_rate=kickoff_live,
        kickoff_live_return=_profile(mean=26.0),
    )


def test_explicit_channel_priors_override_legacy_return_semantics():
    priors = channel_priors_from_policy_row(
        {
            "kickoff_touchback_rate": 0.05,
            "kickoff_touchback_rate_explicit": 0.31,
            "punt_live_return_rate_after_touchback": 0.47,
            "kickoff_live_return_rate_after_touchback": 0.91,
            "fumble_recovery_return_mean": 4.25,
            "fumble_recovery_20_plus_rate": 0.08,
            "fumble_recovery_40_plus_rate": 0.02,
            "fumble_recovery_60_plus_rate": 0.01,
            "fumble_recovery_80_plus_rate": 0.004,
        }
    )
    assert priors.kickoff_touchback_rate == 0.31
    assert priors.punt_live_return_rate == 0.47
    assert priors.kickoff_live_return_rate == 0.91
    assert priors.fumble_recovery.mean == 4.25
    assert not hasattr(priors, "touchdown_rate")


def test_punt_return_selection_precedes_distance_sampling():
    rng = np.random.default_rng(63201)
    priors = _priors(punt_live=0.0, kickoff_touchback=0.0, kickoff_live=1.0)
    events = [simulate_punt_v632(rng, priors=priors) for _ in range(500)]
    eligible = [event for event in events if not event.blocked and not event.touchback and not event.muffed]
    assert eligible
    assert all(event.return_yards == 0.0 for event in eligible)
    assert all(event.fair_catch for event in eligible)


def test_live_punt_return_uses_conditional_distance_population():
    rng = np.random.default_rng(63202)
    priors = _priors(punt_live=1.0, kickoff_touchback=0.0, kickoff_live=1.0)
    events = [simulate_punt_v632(rng, priors=priors) for _ in range(500)]
    eligible = [event for event in events if not event.blocked and not event.touchback and not event.muffed]
    assert eligible
    assert np.mean([event.return_yards for event in eligible]) > 8.0
    assert sum(event.fair_catch for event in eligible) == 0


def test_explicit_kickoff_touchback_gate_precedes_return_distance():
    rng = np.random.default_rng(63203)
    priors = _priors(punt_live=1.0, kickoff_touchback=0.999, kickoff_live=1.0)
    events = [simulate_kickoff_v632(rng, priors=priors) for _ in range(300)]
    assert np.mean([event.touchback for event in events]) > 0.98
    assert all(event.return_yards == 0.0 for event in events if event.touchback)
