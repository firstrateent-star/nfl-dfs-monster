from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreezeEvidence:
    market_blind: bool
    current_state_fresh: bool
    drive_survival_passed: bool
    scoring_state_passed: bool
    clock_possession_passed: bool
    availability_promoted: bool
    resolution_promoted: bool
    conservation_passed: bool
    multi_seed_stable: bool
    shadow_layers_zero_authority: bool
    source_manifest_complete: bool


@dataclass(frozen=True)
class FreezeDecision:
    allowed: bool
    blockers: tuple[str, ...]


def evaluate_football_freeze(evidence: FreezeEvidence) -> FreezeDecision:
    """Decide whether a football-reality state may be canonically frozen.

    This gate is deliberately stricter than ordinary CI. A software-green branch is not a
    football freeze. Every upstream football organ must be evidence-gated, current-state inputs
    must be fresh, and all unpromoted experimental layers must remain zero-authority.
    """
    checks = {
        "market_blind": evidence.market_blind,
        "current_state_fresh": evidence.current_state_fresh,
        "drive_survival": evidence.drive_survival_passed,
        "scoring_state": evidence.scoring_state_passed,
        "clock_possession": evidence.clock_possession_passed,
        "availability": evidence.availability_promoted,
        "resolution": evidence.resolution_promoted,
        "conservation": evidence.conservation_passed,
        "multi_seed_stability": evidence.multi_seed_stable,
        "shadow_isolation": evidence.shadow_layers_zero_authority,
        "source_manifest": evidence.source_manifest_complete,
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    return FreezeDecision(allowed=not blockers, blockers=blockers)
