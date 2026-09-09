import polars as pl

from monster.ingest.madden_defense_special import (
    attach_madden_defense_special_traits,
    compile_madden_defense_special_traits,
)


def _ratings():
    return pl.DataFrame(
        {
            "full_name": ["Edge Rusher", "Cover Corner", "Ace Kicker"],
            "power_moves_rating": [92, 30, 10],
            "finesse_moves_rating": [90, 35, 10],
            "block_shedding_rating": [88, 55, 10],
            "pursuit_rating": [91, 82, 20],
            "man_coverage_rating": [40, 94, 10],
            "zone_coverage_rating": [45, 92, 10],
            "press_rating": [50, 93, 10],
            "play_recognition_rating": [88, 90, 20],
            "tackle_rating": [87, 78, 20],
            "hit_power_rating": [89, 62, 20],
            "speed_rating": [86, 95, 70],
            "acceleration_rating": [88, 96, 72],
            "awareness_rating": [85, 91, 80],
            "kick_power_rating": [20, 20, 97],
            "kick_accuracy_rating": [20, 20, 95],
            "kick_return_rating": [20, 75, 30],
        }
    )


def test_defensive_traits_preserve_position_jurisdiction_signal():
    traits = compile_madden_defense_special_traits(_ratings())
    edge = traits.filter(pl.col("_madden_ds_name") == "edgerusher").row(0, named=True)
    corner = traits.filter(pl.col("_madden_ds_name") == "covercorner").row(0, named=True)
    assert edge["_ds_pass_rush"] > corner["_ds_pass_rush"]
    assert corner["_ds_coverage"] > edge["_ds_coverage"]


def test_attach_does_not_overwrite_existing_skill_speed():
    personnel = pl.DataFrame(
        {
            "display_name": ["Cover Corner"],
            "position": ["CB"],
            "madden_speed": [99.0],
        }
    )
    out = attach_madden_defense_special_traits(personnel, _ratings())
    assert out["madden_speed"][0] == 99.0
    assert out["madden_coverage"][0] is not None
