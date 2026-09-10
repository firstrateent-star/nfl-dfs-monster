from __future__ import annotations

from monster.audit.freeze_gate import FreezeEvidence, evaluate_football_freeze


def _complete(**changes: bool) -> FreezeEvidence:
    values = {
        "market_blind": True,
        "current_state_fresh": True,
        "drive_survival_passed": True,
        "scoring_state_passed": True,
        "clock_possession_passed": True,
        "availability_promoted": True,
        "resolution_promoted": True,
        "conservation_passed": True,
        "multi_seed_stable": True,
        "shadow_layers_zero_authority": True,
        "source_manifest_complete": True,
    }
    values.update(changes)
    return FreezeEvidence(**values)


def test_freeze_requires_every_upstream_gate() -> None:
    decision = evaluate_football_freeze(_complete())
    assert decision.allowed is True
    assert decision.blockers == ()


def test_shadow_authority_or_stale_state_blocks_freeze() -> None:
    decision = evaluate_football_freeze(
        _complete(current_state_fresh=False, shadow_layers_zero_authority=False)
    )
    assert decision.allowed is False
    assert set(decision.blockers) == {"current_state_fresh", "shadow_isolation"}


def test_market_contamination_blocks_freeze_even_if_football_gates_pass() -> None:
    decision = evaluate_football_freeze(_complete(market_blind=False))
    assert decision.allowed is False
    assert decision.blockers == ("market_blind",)
