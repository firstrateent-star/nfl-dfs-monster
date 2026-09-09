from __future__ import annotations

from dataclasses import replace

import numpy as np

from monster.feature_compile.mechanisms import TeamMechanismInputs, _weather_effect
from monster.snapshot.model import TeamState
from monster.snapshot.player import TeamPlayerPool


def apply_environment(
    team: TeamState,
    pool: TeamPlayerPool,
    inputs: TeamMechanismInputs,
) -> tuple[TeamState, TeamPlayerPool]:
    """Apply environment once at its two football jurisdictions.

    Weather changes team scoring efficiency and pass-opportunity state upstream. It never
    awards fantasy points directly. Dome games are exact environment nulls.
    """
    effect = _weather_effect(inputs)
    compiled_team = replace(team, weather_effect=effect)
    compiled_pool = replace(
        pool,
        neutral_pass_rate=float(np.clip(pool.neutral_pass_rate + effect * 0.35, 0.34, 0.72)),
        targetable_dropback_rate=float(
            np.clip(pool.targetable_dropback_rate + effect * 0.20, 0.84, 0.98)
        ),
    )
    return compiled_team, compiled_pool


def apply_environment_to_pool(
    pool: TeamPlayerPool,
    inputs: TeamMechanismInputs,
) -> TeamPlayerPool:
    """Pool-only adapter for runners that compile TeamState later in the pipeline."""
    effect = _weather_effect(inputs)
    return replace(
        pool,
        neutral_pass_rate=float(np.clip(pool.neutral_pass_rate + effect * 0.35, 0.34, 0.72)),
        targetable_dropback_rate=float(
            np.clip(pool.targetable_dropback_rate + effect * 0.20, 0.84, 0.98)
        ),
    )
