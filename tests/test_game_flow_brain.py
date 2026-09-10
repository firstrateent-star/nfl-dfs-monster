from monster.sim.football_state import FootballState
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_brain import (
    DecisionAdjustment,
    HierarchicalFlowEvidence,
    RateEvidence,
    decide_game_flow,
)


def _flow():
    return derive_game_flow_state(
        FootballState(
            possession="A",
            defense="B",
            away_team_id="A",
            home_team_id="B",
            down=3,
            distance=10.0,
            yardline_100=50.0,
        )
    )


def test_sparse_team_context_shrinks_heavily_to_league() -> None:
    decision = decide_game_flow(
        _flow(),
        HierarchicalFlowEvidence(
            league_context=RateEvidence(0.90, 1000),
            team_context=RateEvidence(0.20, 5),
            shrinkage_samples=80.0,
        ),
    )
    assert 0.84 < decision.dropback_probability < 0.90
    assert decision.trace.team_context_authority < 0.06


def test_large_team_context_can_express_real_style_deviation() -> None:
    decision = decide_game_flow(
        _flow(),
        HierarchicalFlowEvidence(
            league_context=RateEvidence(0.70, 1000),
            team_context=RateEvidence(0.40, 400),
            shrinkage_samples=80.0,
        ),
    )
    assert 0.43 < decision.dropback_probability < 0.48
    assert decision.trace.team_context_authority > 0.80


def test_zero_authority_adjustment_is_exactly_neutral() -> None:
    baseline = decide_game_flow(
        _flow(),
        HierarchicalFlowEvidence(league_context=RateEvidence(0.60, 1000)),
    )
    with_venue_claim = decide_game_flow(
        _flow(),
        HierarchicalFlowEvidence(
            league_context=RateEvidence(0.60, 1000),
            venue=DecisionAdjustment(
                source="home_field_unvalidated",
                dropback_logit_shift=1.0,
                authority=0.0,
            ),
        ),
    )
    assert with_venue_claim.dropback_probability == baseline.dropback_probability
    assert with_venue_claim.trace.venue_logit_shift == 0.0


def test_multiple_authorized_layers_are_bounded() -> None:
    evidence = HierarchicalFlowEvidence(
        league_context=RateEvidence(0.50, 1000),
        personnel=DecisionAdjustment("personnel", 1.0, 1.0),
        opponent=DecisionAdjustment("opponent", 1.0, 1.0),
        environment=DecisionAdjustment("environment", 1.0, 1.0),
        venue=DecisionAdjustment("venue", 1.0, 1.0),
        adaptation=DecisionAdjustment("adaptation", 1.0, 1.0),
    )
    decision = decide_game_flow(_flow(), evidence)
    assert decision.trace.total_logit_shift == 1.10
    assert 0.74 < decision.dropback_probability < 0.76


def test_team_neutral_identity_is_lower_authority_than_context() -> None:
    decision = decide_game_flow(
        _flow(),
        HierarchicalFlowEvidence(
            league_context=RateEvidence(0.90, 1000),
            team_neutral_rate=0.40,
            league_neutral_rate=0.60,
        ),
    )
    assert decision.dropback_probability > 0.85
    assert decision.trace.neutral_identity_logit_shift < 0.0
