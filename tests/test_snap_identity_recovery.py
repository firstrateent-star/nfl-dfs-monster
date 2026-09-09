import polars as pl

from monster.ingest.league import _fill_pfr_ids_from_unique_snap_names


def test_unique_snap_name_recovers_missing_pfr_id_without_overwriting_existing_id():
    roster = pl.DataFrame(
        {
            "display_name": ["Isaac Seumalo", "Known Player"],
            "pfr_id": [None, "Known00"],
        }
    )
    snaps = pl.DataFrame(
        {
            "player": ["Isaac Seumalo", "Known Player"],
            "pfr_player_id": ["SeumIs00", "Other00"],
        }
    )
    out = _fill_pfr_ids_from_unique_snap_names(roster, snaps)
    assert out.get_column("pfr_id").to_list() == ["SeumIs00", "Known00"]
    assert out.get_column("pfr_id_recovered_from_snap_name").to_list() == [True, False]


def test_ambiguous_normalized_name_stays_unresolved():
    roster = pl.DataFrame({"display_name": ["John Smith"], "pfr_id": [None]})
    snaps = pl.DataFrame(
        {
            "player": ["John Smith", "John Smith"],
            "pfr_player_id": ["SmitJo01", "SmitJo02"],
        }
    )
    out = _fill_pfr_ids_from_unique_snap_names(roster, snaps)
    assert out.row(0, named=True)["pfr_id"] is None
    assert out.row(0, named=True)["pfr_id_recovered_from_snap_name"] is False


def test_suffix_and_punctuation_normalization_is_exact_not_fuzzy():
    roster = pl.DataFrame({"display_name": ["Paris Johnson Jr."], "pfr_id": [None]})
    snaps = pl.DataFrame(
        {"player": ["Paris Johnson"], "pfr_player_id": ["JohnPa03"]}
    )
    out = _fill_pfr_ids_from_unique_snap_names(roster, snaps)
    assert out.row(0, named=True)["pfr_id"] == "JohnPa03"
