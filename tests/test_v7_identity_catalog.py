from __future__ import annotations

import pytest

from monster.reality.availability import TeamLiveRoster
from monster.reality.identity_catalog import (
    TeamIdentityCatalog,
    apply_live_roster_identity,
)
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _player(player_id: str, position: str, efficiency: float) -> PlayerIdentity:
    return PlayerIdentity(
        player_id=player_id,
        name=player_id,
        position=position,
        usage_weight=0.5,
        efficiency=efficiency,
    )


def test_qb_substitution_uses_backup_identity_not_starter_clone() -> None:
    starter = _player("qb1", "QB", 1.20)
    backup = _player("qb2", "QB", 0.82)
    rb = _player("rb", "RB", 1.05)
    wr = _player("wr", "WR", 1.10)
    te = _player("te", "TE", 0.96)

    team = TeamIdentity("T", starter, (rb, starter), (wr, te))
    catalog = TeamIdentityCatalog("T", (starter, backup, rb, wr, te))
    roster = TeamLiveRoster(
        team_id="T",
        active_player_ids=("qb2", "rb", "wr", "te"),
        current_qb_id="qb2",
        exited_player_ids=("qb1",),
    )

    live = apply_live_roster_identity(team=team, catalog=catalog, roster=roster)

    assert live.quarterback.player_id == "qb2"
    assert live.quarterback.efficiency == pytest.approx(0.82)
    assert "qb1" not in {player.player_id for player in live.rushers}
    assert {player.player_id for player in live.receivers} == {"wr", "te"}


def test_live_roster_filters_exited_receiver_from_future_snaps() -> None:
    qb = _player("qb", "QB", 1.0)
    rb = _player("rb", "RB", 1.0)
    wr1 = _player("wr1", "WR", 1.0)
    wr2 = _player("wr2", "WR", 1.0)
    team = TeamIdentity("T", qb, (rb, qb), (wr1, wr2))
    catalog = TeamIdentityCatalog("T", (qb, rb, wr1, wr2))
    roster = TeamLiveRoster(
        team_id="T",
        active_player_ids=("qb", "rb", "wr2"),
        current_qb_id="qb",
        exited_player_ids=("wr1",),
    )

    live = apply_live_roster_identity(team=team, catalog=catalog, roster=roster)
    assert tuple(player.player_id for player in live.receivers) == ("wr2",)


def test_live_roster_rejects_missing_backup_identity() -> None:
    qb = _player("qb", "QB", 1.0)
    rb = _player("rb", "RB", 1.0)
    wr = _player("wr", "WR", 1.0)
    team = TeamIdentity("T", qb, (rb, qb), (wr,))
    catalog = TeamIdentityCatalog("T", (qb, rb, wr))
    roster = TeamLiveRoster(
        team_id="T",
        active_player_ids=("missing-qb", "rb", "wr"),
        current_qb_id="missing-qb",
    )

    with pytest.raises(KeyError):
        apply_live_roster_identity(team=team, catalog=catalog, roster=roster)
