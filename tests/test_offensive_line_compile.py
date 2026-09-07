import polars as pl

from monster.feature_compile.offensive_line import (
    attach_current_ol_context,
    compile_current_ol_context,
    compile_historical_ol_outcomes,
)
from monster.ingest.nflverse import _supplement_player_ids


def test_ffverse_id_fallback_fills_missing_pfr_without_overwriting_canonical():
    players = pl.DataFrame(
        {
            "gsis_id": ["g1", "g2"],
            "pfr_id": [None, "Canon01"],
            "display_name": ["Guard One", "Guard Two"],
        }
    )
    ff = pl.DataFrame(
        {
            "gsis_id": ["g1", "g2"],
            "pfr_id": ["Fall01", "Wrong02"],
        }
    )
    out = _supplement_player_ids(players, ff).sort("gsis_id")
    assert out.get_column("pfr_id").to_list() == ["Fall01", "Canon01"]


def test_historical_ol_outcomes_reward_lower_pressure_and_better_rushing():
    rows = []
    for team, sacks, hits, run_epa, run_success, yards in [
        ("A", 1, 2, 0.20, 1, 6),
        ("B", 4, 7, -0.20, 0, 0),
    ]:
        for i in range(10):
            rows.append(
                {
                    "posteam": team,
                    "play_type": "pass",
                    "qb_dropback": 1,
                    "sack": 1 if i < sacks else 0,
                    "qb_hit": 1 if i < hits else 0,
                    "rush_attempt": 0,
                    "epa": 0.0,
                    "success": 0,
                    "yards_gained": 0,
                }
            )
        for _ in range(10):
            rows.append(
                {
                    "posteam": team,
                    "play_type": "run",
                    "qb_dropback": 0,
                    "sack": 0,
                    "qb_hit": 0,
                    "rush_attempt": 1,
                    "epa": run_epa,
                    "success": run_success,
                    "yards_gained": yards,
                }
            )
    out = compile_historical_ol_outcomes(pl.DataFrame(rows))
    a = out.filter(pl.col("team_id") == "A").row(0, named=True)
    b = out.filter(pl.col("team_id") == "B").row(0, named=True)
    assert a["historical_pass_protection_signal"] > b["historical_pass_protection_signal"]
    assert a["historical_run_block_signal"] > b["historical_run_block_signal"]


def test_current_line_shrinks_history_when_continuity_is_low():
    personnel = pl.DataFrame(
        {
            "team_id": ["A", "A", "B", "B"],
            "position_group": ["OL", "OL", "OL", "OL"],
            "projected_offense_snap_share": [0.9, 0.9, 0.9, 0.9],
            "participation_uncertainty": [0.10, 0.10, 0.25, 0.25],
            "snap_games_observed": [6, 6, 0, 0],
            "prior_team_id": ["A", "A", None, None],
        }
    )
    historical = pl.DataFrame(
        {
            "team_id": ["A", "B"],
            "historical_pass_protection_signal": [0.8, 0.8],
            "historical_run_block_signal": [0.6, 0.6],
        }
    )
    context = compile_current_ol_context(personnel, historical)
    a = context.filter(pl.col("team_id") == "A").row(0, named=True)
    b = context.filter(pl.col("team_id") == "B").row(0, named=True)
    assert a["ol_continuity"] == 1.0
    assert b["ol_continuity"] == 0.0
    assert a["observed_pass_block_signal"] > b["observed_pass_block_signal"]
    assert a["ol_context_uncertainty"] < b["ol_context_uncertainty"]


def test_ol_context_only_attaches_blocking_signals_to_ol_players():
    personnel = pl.DataFrame(
        {
            "team_id": ["A", "A"],
            "position_group": ["OL", "WR"],
            "player": ["Guard", "Receiver"],
        }
    )
    context = pl.DataFrame(
        {
            "team_id": ["A"],
            "observed_pass_block_signal": [0.5],
            "observed_run_block_signal": [0.4],
            "ol_continuity": [0.8],
            "ol_context_uncertainty": [0.2],
            "historical_ol_authority": [0.7],
        }
    )
    out = attach_current_ol_context(personnel, context).sort("position_group")
    ol = out.filter(pl.col("position_group") == "OL").row(0, named=True)
    wr = out.filter(pl.col("position_group") == "WR").row(0, named=True)
    assert ol["observed_pass_block_signal"] == 0.5
    assert wr["observed_pass_block_signal"] is None
