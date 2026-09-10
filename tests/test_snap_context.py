from __future__ import annotations

import pytest

from monster.sim.football_state import FootballState
from monster.sim.interaction_runtime import InteractionScale, PlayPhase
from monster.sim.snap_context import SnapInteractionState, SnapParticipant, SnapSide


def _participant(player_id: str, team_id: str, side: SnapSide) -> SnapParticipant:
    return SnapParticipant(
        player_id=player_id,
        team_id=team_id,
        position="WR" if side == SnapSide.OFFENSE else "CB",
        side=side,
    )


def test_snap_context_supports_partial_resolution_without_claiming_full_11v11() -> None:
    snap = SnapInteractionState(
        football=FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
        ),
        offense_team_id="A",
        defense_team_id="B",
        offense_participants=(_participant("wr-1", "A", SnapSide.OFFENSE),),
        defense_participants=(_participant("cb-1", "B", SnapSide.DEFENSE),),
        offense_package="11_personnel",
        defense_package="nickel",
    )

    assert snap.is_full_11v11 is False
    context = snap.interaction_context(
        mechanism="catchpoint",
        phase=PlayPhase.RESOLUTION,
        scale=InteractionScale.ONE_V_ONE,
        participant_ids=frozenset({"wr-1", "cb-1"}),
    )
    assert context.participant_ids == frozenset({"wr-1", "cb-1"})


def test_snap_context_rejects_player_not_selected_on_snap() -> None:
    snap = SnapInteractionState(
        football=FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
        ),
        offense_team_id="A",
        defense_team_id="B",
        offense_participants=(_participant("wr-1", "A", SnapSide.OFFENSE),),
        defense_participants=(_participant("cb-1", "B", SnapSide.DEFENSE),),
    )

    with pytest.raises(ValueError, match="not on this snap"):
        snap.interaction_context(
            mechanism="coverage",
            phase=PlayPhase.DEVELOPMENT,
            scale=InteractionScale.ONE_V_ONE,
            participant_ids=frozenset({"wr-1", "cb-bench"}),
        )


def test_full_11v11_requires_exactly_eleven_selected_players_per_side() -> None:
    offense = tuple(
        _participant(f"off-{idx}", "A", SnapSide.OFFENSE) for idx in range(11)
    )
    defense = tuple(
        _participant(f"def-{idx}", "B", SnapSide.DEFENSE) for idx in range(11)
    )
    snap = SnapInteractionState(
        football=FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
        ),
        offense_team_id="A",
        defense_team_id="B",
        offense_participants=offense,
        defense_participants=defense,
    )

    assert snap.is_full_11v11 is True
