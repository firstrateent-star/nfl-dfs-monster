from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoachingSystemIdentity:
    """Separated coaching/system state for future tactical authority.

    Historical team behavior remains the current production prior. Named coach/coordinator and
    scheme evidence may populate this object later, but cannot alter football worlds until its
    authority is explicitly promoted. This prevents a coaching label from double-counting the
    tendencies already learned from team history.
    """

    team_id: str
    head_coach_id: str | None = None
    offensive_coordinator_id: str | None = None
    defensive_coordinator_id: str | None = None
    offensive_system_id: str | None = None
    defensive_system_id: str | None = None
    continuity: float = 0.5
    offensive_change_uncertainty: float = 0.0
    defensive_change_uncertainty: float = 0.0
    pace_authority: float = 0.0
    play_call_authority: float = 0.0
    fourth_down_authority: float = 0.0
    defensive_tactical_authority: float = 0.0

    @property
    def production_authority(self) -> float:
        return max(
            self.pace_authority,
            self.play_call_authority,
            self.fourth_down_authority,
            self.defensive_tactical_authority,
        )


def shadow_system_identity(
    team_id: str,
    *,
    continuity: float,
    head_coach_id: str | None = None,
    offensive_coordinator_id: str | None = None,
    defensive_coordinator_id: str | None = None,
    offensive_system_id: str | None = None,
    defensive_system_id: str | None = None,
) -> CoachingSystemIdentity:
    """Create a trace-only system identity with uncertainty increasing as continuity falls."""
    continuity = float(np.clip(continuity, 0.0, 1.0))
    change_uncertainty = float(np.clip(1.0 - continuity, 0.0, 1.0))
    return CoachingSystemIdentity(
        team_id=team_id,
        head_coach_id=head_coach_id,
        offensive_coordinator_id=offensive_coordinator_id,
        defensive_coordinator_id=defensive_coordinator_id,
        offensive_system_id=offensive_system_id,
        defensive_system_id=defensive_system_id,
        continuity=continuity,
        offensive_change_uncertainty=change_uncertainty,
        defensive_change_uncertainty=change_uncertainty,
    )
