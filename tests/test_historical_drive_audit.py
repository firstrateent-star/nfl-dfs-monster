from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_audit_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts / "audit_historical_drive_reality.py"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("historical_drive_audit_test_module", script)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load historical drive audit module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


AUDIT = _load_audit_module()


def test_kickoff_cannot_define_drive_start_or_red_zone_snap() -> None:
    rows = [
        {
            "play_type": "kickoff",
            "yardline_100": 35.0,
            "game_seconds_remaining": 3600.0,
            "qtr": 1,
            "qb_dropback": 0,
            "rush_attempt": 0,
            "yards_gained": 0.0,
        },
        {
            "play_type": "pass",
            "yardline_100": 75.0,
            "game_seconds_remaining": 3585.0,
            "qtr": 1,
            "qb_dropback": 1,
            "rush_attempt": 0,
            "yards_gained": 8.0,
            "posteam_score": 0,
            "posteam_score_post": 0,
        },
    ]

    drive = AUDIT._drive_row("game", "1", "A", rows)

    assert drive is not None
    assert drive["start_yardline_100"] == 25.0
    assert drive["start_seconds_remaining"] == 3585.0
    assert drive["end_seconds_remaining"] == 3585.0
    assert drive["red_zone_snap_seen"] is False
    assert drive["overtime"] is False


def test_long_touchdown_reaches_red_zone_without_red_zone_snap() -> None:
    rows = [
        {
            "play_type": "pass",
            "yardline_100": 50.0,
            "game_seconds_remaining": 1200.0,
            "qtr": 3,
            "qb_dropback": 1,
            "rush_attempt": 0,
            "yards_gained": 50.0,
            "pass_touchdown": 1,
            "posteam_score": 0,
            "posteam_score_post": 6,
        }
    ]

    drive = AUDIT._drive_row("game", "2", "A", rows)

    assert drive is not None
    assert drive["red_zone_entered"] is True
    assert drive["red_zone_snap_seen"] is False
    assert drive["terminal"] == "touchdown"
    assert drive["start_seconds_remaining"] == 1200.0
    assert drive["end_seconds_remaining"] == 1200.0


def test_overtime_drive_is_marked_from_first_scrimmage_state() -> None:
    rows = [
        {
            "play_type": "run",
            "yardline_100": 75.0,
            "game_seconds_remaining": 540.0,
            "qtr": 5,
            "qb_dropback": 0,
            "rush_attempt": 1,
            "yards_gained": 4.0,
            "posteam_score": 20,
            "posteam_score_post": 20,
        }
    ]

    drive = AUDIT._drive_row("game", "OT", "A", rows)

    assert drive is not None
    assert drive["overtime"] is True


def test_non_scrimmage_only_group_is_not_definition_safe_drive() -> None:
    rows = [
        {
            "play_type": "field_goal",
            "yardline_100": 18.0,
            "game_seconds_remaining": 600.0,
            "qtr": 4,
            "qb_dropback": 0,
            "rush_attempt": 0,
            "field_goal_attempt": 1,
            "field_goal_result": "made",
            "posteam_score": 0,
            "posteam_score_post": 3,
        }
    ]

    assert AUDIT._drive_row("game", "3", "A", rows) is None
