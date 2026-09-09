from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TeamState:
    team_id: str
    opponent_id: str
    offense_strength: float = 0.0
    defense_strength: float = 0.0
    neutral_pass_rate: float = 0.56
    pace_factor: float = 1.0
    drives_per_game: float = 10.5
    td_drive_rate: float = 0.22
    fg_drive_rate: float = 0.14
    turnover_drive_rate: float = 0.11
    red_zone_td_rate: float = 0.55
    offensive_epa_per_play: float = 0.0
    offensive_success_rate: float = 0.44
    offensive_explosive_rate: float = 0.10
    defensive_td_drive_rate_allowed: float = 0.22
    defensive_fg_drive_rate_allowed: float = 0.14
    defensive_takeaway_drive_rate: float = 0.11
    defensive_epa_allowed_per_play: float = 0.0
    defensive_explosive_rate_allowed: float = 0.10
    defensive_sack_rate: float = 0.07
    defensive_qb_hit_rate: float = 0.18

    # Snap-weighted all-player unit mechanisms. These are compiled from offense,
    # defense and special-teams personnel before the game is simulated.
    pass_protection_effect: float = 0.0
    run_block_effect: float = 0.0
    pass_rush_effect: float = 0.0
    coverage_effect: float = 0.0
    run_defense_effect: float = 0.0
    special_teams_effect: float = 0.0

    # Neutral defaults preserve the pre-OL simulator when no strengthened context is
    # supplied. Current individual capability can move the mean; measured continuity
    # and uncertainty are injected only by the league context compiler.
    offensive_line_continuity: float = 1.0
    offensive_line_uncertainty: float = 0.0

    continuity: float = 0.5
    injury_effect: float = 0.0
    weather_effect: float = 0.0
    physical_madden_effect: float = 0.0
    coaching_entropy: float = 0.10
    uncertainty: float = 0.10


@dataclass(frozen=True)
class GameState:
    game_id: str
    away: TeamState
    home: TeamState
    dome: bool = False
    feature_names: frozenset[str] = field(default_factory=frozenset)
