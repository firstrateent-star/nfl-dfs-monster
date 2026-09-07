from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    display_name: str
    position: str
    team_id: str
    target_share: float = 0.0
    rush_share: float = 0.0
    red_zone_target_share: float = 0.0
    red_zone_rush_share: float = 0.0
    receiving_td_share: float = 0.0
    rushing_td_share: float = 0.0
    catch_rate: float = 0.65
    yards_per_reception: float = 10.5
    yards_per_carry: float = 4.2
    active_probability: float = 1.0
    effectiveness_if_active: float = 1.0
    role_uncertainty: float = 0.08
    explosive_modifier: float = 1.0
    catchpoint_modifier: float = 1.0
    rushing_efficiency_modifier: float = 1.0
    qb_pass_share: float = 0.0


@dataclass(frozen=True)
class TeamPlayerPool:
    team_id: str
    players: tuple[PlayerState, ...]
    neutral_pass_rate: float = 0.56
    plays_per_drive: float = 6.1
    pass_td_share: float = 0.64
    sack_rate: float = 0.065
    targetable_dropback_rate: float = 0.94
    script_pass_sensitivity: float = 0.0035
    play_volume_uncertainty: float = 0.06
