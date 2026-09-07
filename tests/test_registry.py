from monster.registry import FeatureRegistry


def test_market_features_are_forbidden_upstream():
    r = FeatureRegistry.load()
    assert "sportsbook_total" in r.forbidden_upstream()
    assert "sportsbook_spread" in r.forbidden_upstream()
    assert "dfs_ownership" in r.forbidden_upstream()
    assert "height_in" in r.blind_allowed()
    assert "madden_speed" in r.blind_allowed()
