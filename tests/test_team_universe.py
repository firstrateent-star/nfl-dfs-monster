from monster.teams import NFL_TEAMS, normalize_team_id


def test_monster_has_all_32_current_nfl_teams():
    assert len(NFL_TEAMS) == 32
    assert len(set(NFL_TEAMS)) == 32


def test_common_source_aliases_normalize_to_monster_ids():
    assert normalize_team_id("JAX") == "JAC"
    assert normalize_team_id("LA") == "LAR"
    assert normalize_team_id("SFO") == "SF"
    assert normalize_team_id("LVR") == "LV"
    assert normalize_team_id("WSH") == "WAS"


def test_unknown_team_identifier_fails_loudly():
    try:
        normalize_team_id("XYZ")
    except ValueError:
        return
    raise AssertionError("Unknown team IDs must not silently enter canonical Monster state")
