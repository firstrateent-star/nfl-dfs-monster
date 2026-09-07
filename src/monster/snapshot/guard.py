from __future__ import annotations
from monster.registry import FeatureRegistry

class MarketLeakageError(RuntimeError):
    pass


def assert_market_blind(feature_names: set[str], registry: FeatureRegistry) -> None:
    forbidden = feature_names.intersection(registry.forbidden_upstream())
    if forbidden:
        raise MarketLeakageError(f"Market-derived features present upstream: {sorted(forbidden)}")
