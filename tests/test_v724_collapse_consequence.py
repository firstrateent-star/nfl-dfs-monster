from __future__ import annotations

from monster.reality import failure_paths_v722 as v722
from monster.reality import failure_paths_v723 as v723
from monster.reality import failure_paths_v724 as v724
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


def _state() -> FootballState:
    return FootballState(
        possession="A",
        defense="B",
        quarter=3,
        seconds_remaining=600,
        yardline_100=90.0,
        down=3,
        distance=5.0,
        away_score=10,
        home_score=17,
        away_team_id="A",
        home_team_id="B",
    )


def _world(seed: int):
    v722.reset_failure_paths_v722()
    v723.reset_v723_state()
    v724.reset_v724_state()
    v722.begin_failure_path_world_v722(
        seed=seed,
        game="A@B",
        team_ids=("A", "B"),
    )
    v723.recalibrate_current_failure_world_v723(
        seed=seed,
        game="A@B",
        team_ids=("A", "B"),
    )
    v724.reweight_finishing_consequence_v724(seed=seed, game="A@B")
    return v722.current_failure_world_v722()


def test_v724_preserves_v723_tail_mass() -> None:
    hard = 0
    affected = 0
    samples = 2000
    for seed in range(724000, 724000 + samples):
        world = _world(seed)
        assert world is not None
        mode = world.teams["A"].collapse_mode
        hard += int(mode == "hard")
        affected += int(mode != "none")
    assert 0.055 <= hard / samples <= 0.095
    assert 0.20 <= affected / samples <= 0.28


def test_v724_finishing_consequence_is_mode_ordered() -> None:
    world = _world(724001)
    assert world is not None
    team = world.teams["A"]

    world.game_drag = 0.0
    team.collapse_mode = "none"
    team.collapse_strength = 0.0
    v724.reweight_finishing_consequence_v724(seed=724001, game="A@B")
    normal = team.finishing_friction

    team.collapse_mode = "drag"
    team.collapse_strength = 0.12
    v724.reweight_finishing_consequence_v724(seed=724001, game="A@B")
    drag = team.finishing_friction

    team.collapse_mode = "hard"
    team.collapse_strength = 0.26
    v724.reweight_finishing_consequence_v724(seed=724001, game="A@B")
    hard = team.finishing_friction

    assert normal <= 0.024
    assert drag >= 0.19
    assert hard >= 0.42
    assert normal < drag < hard


def test_v724_hard_world_has_materially_larger_stall_probability() -> None:
    world = _world(724002)
    assert world is not None
    team = world.teams["A"]
    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="qb1",
        target_id="wr1",
        pass_result=PassResult.COMPLETE,
        yards=10.0,
        touchdown=True,
    )

    world.game_drag = 0.0
    team.collapse_mode = "none"
    team.finishing_friction = 0.02
    normal = v724._stall_probability_v724(state=_state(), team=team, event=event)

    team.collapse_mode = "drag"
    team.finishing_friction = 0.24
    drag = v724._stall_probability_v724(state=_state(), team=team, event=event)

    team.collapse_mode = "hard"
    team.finishing_friction = 0.50
    hard = v724._stall_probability_v724(state=_state(), team=team, event=event)

    assert normal < 0.02
    assert 0.20 < drag < 0.40
    assert hard > 0.50
    assert normal < drag < hard


def test_v724_existing_turnover_remains_ineligible_for_stall() -> None:
    world = _world(724003)
    assert world is not None
    team = world.teams["A"]
    team.collapse_mode = "hard"
    team.finishing_friction = 0.64
    event = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=7,
        rusher_id="rb1",
        fumbler_id="rb1",
        yards=12.0,
        turnover=True,
    )
    assert v724._stall_probability_v724(state=_state(), team=team, event=event) == 0.0
