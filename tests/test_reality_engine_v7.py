from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from monster.reality.engine import RealityGameRequest, V6CompatibilityEngine
from monster.reality.ledger import (
    LedgerEventKind,
    LedgerFidelity,
    ParticipantSnapshot,
)
from monster.reality.world_state import GameDayLatents
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.12)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.88)
    wr = PlayerIdentity(f"{team}-wr", f"{team} WR", "WR", usage_weight=0.55)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.45)
    return TeamIdentity(team, qb, (rb, qb), (wr, te))


def test_v7_compatibility_engine_has_zero_football_drift() -> None:
    away = _team("AWY")
    home = _team("HME")
    seed = 7001001

    direct = simulate_game(away, home, seed=seed)
    request = RealityGameRequest.compatibility(away=away, home=home, seed=seed)
    wrapped = V6CompatibilityEngine().run(request)

    assert wrapped.football.final_state == direct.final_state
    assert wrapped.football.plays == direct.plays
    assert wrapped.football.player_stats == direct.player_stats
    assert wrapped.football.drive_traces == direct.drive_traces
    assert wrapped.football.special_teams_events == direct.special_teams_events
    assert wrapped.football.return_events == direct.return_events
    assert wrapped.football.try_events == direct.try_events


def test_v7_ledger_is_deterministic_and_closes_to_same_scoreboard() -> None:
    away = _team("AWY")
    home = _team("HME")
    request = RealityGameRequest.compatibility(away=away, home=home, seed=7001002)

    first = V6CompatibilityEngine().run(request)
    second = V6CompatibilityEngine().run(request)

    assert first.ledger.digest() == second.ledger.digest()
    assert len(first.ledger.snap_records) == len(first.football.plays)
    assert len(first.ledger.drive_records) == len(first.football.drive_traces)

    final = first.ledger.records[-1]
    assert final.event_kind == LedgerEventKind.GAME_FINAL
    assert final.fidelity == LedgerFidelity.V6_ADAPTER
    assert final.post_state is not None
    assert final.post_state.away_score == first.football.final_state.away_score
    assert final.post_state.home_score == first.football.final_state.home_score


def test_v7_runtime_constitution_is_market_blind() -> None:
    result = V6CompatibilityEngine().run(
        RealityGameRequest.compatibility(
            away=_team("AWY"),
            home=_team("HME"),
            seed=7001003,
        )
    )

    assert result.runtime.football_first
    assert not result.runtime.market_inputs_to_football
    assert not result.runtime.direct_fantasy_inputs_to_football
    assert not result.runtime.direct_score_adjustment
    assert result.runtime.runtime_hash


def test_world_and_ledger_contracts_are_immutable() -> None:
    request = RealityGameRequest.compatibility(
        away=_team("AWY"),
        home=_team("HME"),
        seed=7001004,
        latents=GameDayLatents.from_mapping(
            {"qb_execution_day": 0.4, "pass_protection_day": -0.2},
            source="test",
        ),
    )
    result = V6CompatibilityEngine().run(request)

    assert request.world.latents.value("qb_execution_day") == pytest.approx(0.4)
    with pytest.raises(FrozenInstanceError):
        request.world.key.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.ledger.records[0].sequence = 99  # type: ignore[misc]


def test_participant_snapshot_knows_exact_11v11() -> None:
    participants = ParticipantSnapshot(
        offense_player_ids=tuple(f"o{i}" for i in range(11)),
        defense_player_ids=tuple(f"d{i}" for i in range(11)),
    )
    assert participants.exact_11v11

    with pytest.raises(ValueError):
        ParticipantSnapshot(
            offense_player_ids=("duplicate", "duplicate"),
            defense_player_ids=(),
        )


def test_v6_adapter_does_not_claim_native_state_fidelity() -> None:
    result = V6CompatibilityEngine().run(
        RealityGameRequest.compatibility(
            away=_team("AWY"),
            home=_team("HME"),
            seed=7001005,
        )
    )

    assert result.ledger.snap_records
    assert all(
        record.fidelity == LedgerFidelity.V6_ADAPTER
        for record in result.ledger.snap_records
    )
    assert all(record.pre_state is None for record in result.ledger.snap_records)
