from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from monster.sim.football_state import FootballState
from monster.sim.interaction_runtime import (
    InteractionContext,
    InteractionScale,
    PlayPhase,
)


class SnapSide(StrEnum):
    OFFENSE = "offense"
    DEFENSE = "defense"


@dataclass(frozen=True)
class SnapParticipant:
    """Identity/role information for one player selected onto a simulated snap.

    Performance evidence is intentionally not stored here; metrics travel through the
    governed interaction runtime so identity and evidence cannot become accidentally fused.
    """

    player_id: str
    team_id: str
    position: str
    side: SnapSide
    depth_rank: int | None = None
    role: str | None = None


@dataclass(frozen=True)
class SnapInteractionState:
    """Common state seam for progressive 1v1 -> 11v11 simulation resolution."""

    football: FootballState
    offense_team_id: str
    defense_team_id: str
    offense_participants: tuple[SnapParticipant, ...] = ()
    defense_participants: tuple[SnapParticipant, ...] = ()
    offense_package: str | None = None
    defense_package: str | None = None

    @property
    def participant_ids(self) -> frozenset[str]:
        return frozenset(
            player.player_id
            for player in (*self.offense_participants, *self.defense_participants)
        )

    @property
    def is_full_11v11(self) -> bool:
        return len(self.offense_participants) == 11 and len(self.defense_participants) == 11

    def interaction_context(
        self,
        *,
        mechanism: str,
        phase: PlayPhase,
        scale: InteractionScale,
        participant_ids: frozenset[str] | None = None,
    ) -> InteractionContext:
        participants = self.participant_ids if participant_ids is None else participant_ids
        if not participants.issubset(self.participant_ids):
            unknown = sorted(participants.difference(self.participant_ids))
            raise ValueError(f"Interaction references players not on this snap: {unknown}")
        return InteractionContext(
            mechanism=mechanism,
            phase=phase,
            scale=scale,
            participant_ids=participants,
        )
