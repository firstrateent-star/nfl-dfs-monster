import polars as pl

from monster.ingest.madden_players import attach_madden_ol_ratings, compile_madden_ol_proxies


def _ratings():
    return pl.DataFrame(
        {
            "full_name": ["Elite Tackle", "Average Guard"],
            "position": ["LT", "RG"],
            "team_name": ["Buffalo Bills", "Chicago Bears"],
            "pass_block_rating": [96, 78],
            "pass_block_power_rating": [94, 78],
            "pass_block_finesse_rating": [98, 78],
            "run_block_rating": [90, 78],
            "run_block_power_rating": [92, 78],
            "run_block_finesse_rating": [88, 78],
            "impact_block_rating": [95, 78],
            "awareness_rating": [96, 78],
            "strength_rating": [94, 78],
            "injury_rating": [90, 80],
            "stamina_rating": [92, 80],
        }
    )


def test_madden_ol_proxy_is_shrunk_toward_neutral():
    out = compile_madden_ol_proxies(_ratings())
    elite = out.filter(pl.col("full_name") == "Elite Tackle").row(0, named=True)
    assert elite["madden_pass_block_composite_raw"] > 90
    assert 78 < elite["madden_pass_block"] < elite["madden_pass_block_composite_raw"]


def test_madden_attachment_prefers_current_team_but_allows_unique_name_trade_fallback():
    personnel = pl.DataFrame(
        {
            "team_id": ["BUF", "MIA", "CHI"],
            "display_name": ["Elite Tackle", "Elite Tackle", "Average Guard"],
            "position_group": ["OL", "OL", "OL"],
        }
    )
    out = attach_madden_ol_ratings(personnel, _ratings())
    buf = out.filter(pl.col("team_id") == "BUF").row(0, named=True)
    mia = out.filter(pl.col("team_id") == "MIA").row(0, named=True)
    chi = out.filter(pl.col("team_id") == "CHI").row(0, named=True)
    assert buf["madden_match_type"] == "team_name"
    assert mia["madden_match_type"] == "unique_name"
    assert chi["madden_pass_block"] is not None


def test_non_ol_never_gets_ol_blocking_authority():
    personnel = pl.DataFrame(
        {
            "team_id": ["BUF"],
            "display_name": ["Elite Tackle"],
            "position_group": ["WR"],
        }
    )
    out = attach_madden_ol_ratings(personnel, _ratings())
    assert out.row(0, named=True)["madden_pass_block"] is None
