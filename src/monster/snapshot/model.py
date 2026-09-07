from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class TeamState:
    team_id: str
    opponent_id: str
    offense_strength: float
    defense_strength: float
    neutral_pass_rate: float
    pace_factor: float
    continuity: float = 0.5
    injury_effect: float = 0.0
    weather_effect: float = 0.0
    uncertainty: float = 0.10

@dataclass(frozen=True)
class GameState:
    game_id: str
    away: TeamState
    home: TeamState
    dome: bool = False
    feature_names: frozenset[str] = field(default_factory=frozenset)
