import pytest
from monster.registry import FeatureRegistry
from monster.snapshot.guard import assert_market_blind, MarketLeakageError

def test_guard_accepts_football_features():
    r = FeatureRegistry.load()
    assert_market_blind({"height_in", "neutral_pass_rate", "wind_mph"}, r)

def test_guard_rejects_market():
    r = FeatureRegistry.load()
    with pytest.raises(MarketLeakageError):
        assert_market_blind({"height_in", "sportsbook_total"}, r)
