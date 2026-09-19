"""MONSTER v7 Reality Engine contracts."""

from monster.reality.availability import (
    AvailabilityTransition,
    ExitReason,
    GameAvailabilityState,
    TeamLiveRoster,
)
from monster.reality.engine import (
    RealityEngine,
    RealityGameRequest,
    RealityGameResult,
    V6CompatibilityEngine,
)
from monster.reality.game_latents import (
    sample_game_day_latents,
    team_latent_vector,
)
from monster.reality.identity_catalog import (
    TeamIdentityCatalog,
    apply_live_roster_identity,
    compile_team_identity_catalog,
)
from monster.reality.latent_authority import (
    LATENT_AUTHORITY,
    LatentAuthoritySpec,
    LatentMechanism,
    RoutedLatent,
    route_team_latent,
)
from monster.reality.ledger import (
    LedgerEventKind,
    LedgerFidelity,
    LedgerRecord,
    ParticipantSnapshot,
    RealityLedger,
)
from monster.reality.native_ledger import NativeRealityLedgerBuilder
from monster.reality.qb import (
    QBRushFamily,
    QBRushFamilySummary,
    classify_qb_rush_event,
    summarize_qb_rush_families,
)
from monster.reality.reducers import final_score, reduce_player_box_scores
from monster.reality.world_state import (
    GameDayLatents,
    LatentFactor,
    PregameWorld,
    TeamPregameState,
    WorldKey,
)

__all__ = [
    "LATENT_AUTHORITY",
    "AvailabilityTransition",
    "ExitReason",
    "GameAvailabilityState",
    "GameDayLatents",
    "LatentAuthoritySpec",
    "LatentFactor",
    "LatentMechanism",
    "LedgerEventKind",
    "LedgerFidelity",
    "LedgerRecord",
    "NativeRealityLedgerBuilder",
    "ParticipantSnapshot",
    "PregameWorld",
    "QBRushFamily",
    "QBRushFamilySummary",
    "RealityEngine",
    "RealityGameRequest",
    "RealityGameResult",
    "RealityLedger",
    "RoutedLatent",
    "TeamIdentityCatalog",
    "TeamLiveRoster",
    "TeamPregameState",
    "V6CompatibilityEngine",
    "WorldKey",
    "apply_live_roster_identity",
    "classify_qb_rush_event",
    "compile_team_identity_catalog",
    "final_score",
    "reduce_player_box_scores",
    "route_team_latent",
    "sample_game_day_latents",
    "summarize_qb_rush_families",
    "team_latent_vector",
]
