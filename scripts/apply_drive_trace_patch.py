from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_nth(text: str, old: str, new: str, *, occurrence: int, label: str) -> str:
    start = -1
    for _ in range(occurrence):
        start = text.find(old, start + 1)
        if start < 0:
            raise RuntimeError(f"{label}: occurrence {occurrence} not found")
    return text[:start] + new + text[start + len(old) :]


# A safety is its own possession terminal; do not hide it under turnover.
state_path = Path("src/monster/sim/football_state.py")
state = state_path.read_text()
state = replace_once(
    state,
    '    MISSED_FIELD_GOAL = "missed_field_goal"\n    PUNT = "punt"\n',
    '    MISSED_FIELD_GOAL = "missed_field_goal"\n    SAFETY = "safety"\n    PUNT = "punt"\n',
    label="safety terminal",
)
state_path.write_text(state)


# Let the passive observer notice penalty-only state changes and know whether a drive had activity.
trace_path = Path("src/monster/sim/drive_trace.py")
trace = trace_path.read_text()
trace = replace_once(
    trace,
    "        self.last_offense_yardline = start.yardline_100\n",
    "        self.last_offense_yardline = start.yardline_100\n        self.observed_events = 0\n",
    label="drive trace activity init",
)
trace = replace_once(
    trace,
    '        if before.possession != self.start.possession:\n            raise ValueError("drive observer received an event from a different offense")\n        if event.play_type not in {PlayType.RUN, PlayType.PASS}:\n',
    '        if before.possession != self.start.possession:\n            raise ValueError("drive observer received an event from a different offense")\n        self.observed_events += 1\n        if event.play_type not in {PlayType.RUN, PlayType.PASS}:\n',
    label="drive trace activity observe",
)
trace = replace_once(
    trace,
    "    def finish(\n",
    '''    @property\n    def has_activity(self) -> bool:\n        return self.observed_events > 0\n\n    def observe_penalty(self, before: FootballState, after: FootballState) -> None:\n        \"\"\"Observe an enforced penalty without counting it as a scrimmage play.\"\"\"\n        if before.possession != self.start.possession:\n            raise ValueError(\"drive observer received a penalty from a different offense\")\n        self.observed_events += 1\n        if after.possession == self.start.possession:\n            self.last_offense_yardline = after.yardline_100\n            self.red_zone_entered = self.red_zone_entered or after.yardline_100 >= 80.0\n            self.goal_to_go_reached = self.goal_to_go_reached or _goal_to_go(after)\n\n    def finish(\n''',
    label="drive trace penalty observer",
)
trace_path.write_text(trace)


loop_path = Path("src/monster/sim/game_loop_v13.py")
loop = loop_path.read_text()
loop = replace_once(
    loop,
    "import numpy as np\n\nfrom monster.sim.clock import (\n",
    "import numpy as np\n\nfrom monster.sim.clock import (\n",
    label="stable numpy import anchor",
)
loop = replace_once(
    loop,
    ")\nfrom monster.sim.football_state import (\n    FootballState,\n",
    ")\nfrom monster.sim.drive_trace import DriveTrace, DriveTraceRecorder\nfrom monster.sim.football_state import (\n    FootballState,\n    PossessionTerminal,\n",
    label="drive trace imports",
)
loop = replace_once(
    loop,
    "    player_stats: dict[str, PlayerBoxScore]\n    drives: int\n    defensive_stats: dict[str, DefensiveBoxScore] | None = None\n",
    "    player_stats: dict[str, PlayerBoxScore]\n    drives: int\n    drive_traces: tuple[DriveTrace, ...] = ()\n    defensive_stats: dict[str, DefensiveBoxScore] | None = None\n",
    label="game result drive traces",
)
loop = replace_once(
    loop,
    "    drives = 1\n    safeties = 0\n",
    "    drives = 1\n    drive_traces: list[DriveTrace] = []\n    drive_recorder = DriveTraceRecorder(state)\n    safeties = 0\n",
    label="regulation recorder init",
)
loop = replace_nth(
    loop,
    "            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)\n",
    "            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)\n            drive_recorder.observe_penalty(before, state)\n",
    occurrence=1,
    label="regulation penalty observation",
)
loop = replace_nth(
    loop,
    "            _record(event, stats, defensive_stats)\n",
    "            _record(event, stats, defensive_stats)\n            drive_recorder.observe(before, event)\n",
    occurrence=1,
    label="regulation event observation",
)

# Regulation terminals.
loop = replace_once(
    loop,
    "                drives += 1\n            elif event.play_type == PlayType.FIELD_GOAL:\n",
    "                drive_traces.append(drive_recorder.finish(state, PossessionTerminal.PUNT, points=0))\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            elif event.play_type == PlayType.FIELD_GOAL:\n",
    label="regulation punt terminal",
)
loop = replace_once(
    loop,
    '''                state = advance_game_clock(state, event.elapsed_seconds)\n                if fg.made:\n                    state = _kickoff(\n                        _add_score(state, 3, away_team_id=away.team_id), rng, special\n                    )\n                else:\n                    state = missed_field_goal_transition(state, elapsed_seconds=0)\n                drives += 1\n            elif is_safety(yardline_100=state.yardline_100, yards=event.yards):\n''',
    '''                state = advance_game_clock(state, event.elapsed_seconds)\n                if fg.made:\n                    scored_state = _add_score(state, 3, away_team_id=away.team_id)\n                    drive_traces.append(\n                        drive_recorder.finish(scored_state, PossessionTerminal.FIELD_GOAL)\n                    )\n                    state = _kickoff(scored_state, rng, special)\n                else:\n                    state = missed_field_goal_transition(state, elapsed_seconds=0)\n                    drive_traces.append(\n                        drive_recorder.finish(state, PossessionTerminal.MISSED_FIELD_GOAL, points=0)\n                    )\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            elif is_safety(yardline_100=state.yardline_100, yards=event.yards):\n''',
    label="regulation field goal terminal",
)
loop = replace_once(
    loop,
    '''                safeties += 1\n                state = _kickoff(state, rng, special)\n                drives += 1\n            elif event.turnover:\n''',
    '''                safeties += 1\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.SAFETY, points=0)\n                )\n                state = _kickoff(state, rng, special)\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            elif event.turnover:\n''',
    label="regulation safety terminal",
)
loop = replace_once(
    loop,
    '''                state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)\n                drives += 1\n            elif event.touchdown:\n''',
    '''                state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.TURNOVER, points=0)\n                )\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            elif event.touchdown:\n''',
    label="regulation turnover terminal",
)
loop = replace_once(
    loop,
    '''                tries.append(trial)\n                state = _add_score(state, trial.points, away_team_id=away.team_id)\n                state = _kickoff(state, rng, special)\n                drives += 1\n            else:\n''',
    '''                tries.append(trial)\n                state = _add_score(state, trial.points, away_team_id=away.team_id)\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN)\n                )\n                state = _kickoff(state, rng, special)\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            else:\n''',
    label="regulation touchdown terminal",
)
loop = replace_once(
    loop,
    '''                if before.down == 4 and event.yards < before.distance:\n                    state = turnover_on_downs(state)\n                    drives += 1\n        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:\n''',
    '''                if before.down == 4 and event.yards < before.distance:\n                    state = turnover_on_downs(state)\n                    drive_traces.append(\n                        drive_recorder.finish(state, PossessionTerminal.TURNOVER_ON_DOWNS, points=0)\n                    )\n                    drives += 1\n                    drive_recorder = DriveTraceRecorder(state)\n        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:\n''',
    label="regulation downs terminal",
)
loop = replace_once(
    loop,
    '''        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:\n            halftime_done = True\n            state = FootballState(\n''',
    '''        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:\n            if drive_recorder.has_activity and drive_recorder.start.possession == before.possession:\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.HALFTIME)\n                )\n            halftime_done = True\n            state = FootballState(\n''',
    label="halftime terminal",
)
loop = replace_once(
    loop,
    '''            )\n            drives += 1\n    if state.seconds_remaining > 0:\n        state = replace(state, seconds_remaining=0, quarter=4)\n    return GameResultV13(\n''',
    '''            )\n            drives += 1\n            drive_recorder = DriveTraceRecorder(state)\n    if state.seconds_remaining > 0:\n        state = replace(state, seconds_remaining=0, quarter=4)\n    if drive_recorder.has_activity:\n        drive_traces.append(drive_recorder.finish(state, PossessionTerminal.END_GAME))\n    return GameResultV13(\n''',
    label="regulation end terminal",
)
loop = replace_nth(
    loop,
    "        drives=drives,\n        defensive_stats=defensive_stats,\n",
    "        drives=drives,\n        drive_traces=tuple(drive_traces),\n        defensive_stats=defensive_stats,\n",
    occurrence=1,
    label="regulation result traces",
)

# Overtime starts with the regulation traces and a new possession after the opening kickoff.
loop = replace_once(
    loop,
    "    drives = regulation.drives\n    safeties = regulation.safeties\n",
    "    drives = regulation.drives\n    drive_traces = list(regulation.drive_traces)\n    safeties = regulation.safeties\n",
    label="overtime trace copy",
)
loop = replace_once(
    loop,
    "    state = _kickoff(state, rng, special)\n    drives += 1\n    initial_completed: set[str] = set()\n",
    "    state = _kickoff(state, rng, special)\n    drives += 1\n    drive_recorder = DriveTraceRecorder(state)\n    initial_completed: set[str] = set()\n",
    label="overtime recorder init",
)
loop = replace_nth(
    loop,
    "            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)\n            continue\n",
    "            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)\n            drive_recorder.observe_penalty(before, state)\n            continue\n",
    occurrence=1,
    label="overtime penalty observation",
)
loop = replace_nth(
    loop,
    "        _record(event, stats, defensive_stats)\n",
    "        _record(event, stats, defensive_stats)\n        drive_recorder.observe(before, event)\n",
    occurrence=1,
    label="overtime event observation",
)
loop = replace_once(
    loop,
    '''            drives += 1\n            initial_completed.add(offense.team_id)\n            if len(initial_completed) >= 2 and state.away_score != state.home_score:\n                break\n            continue\n\n        if event.play_type == PlayType.FIELD_GOAL:\n''',
    '''            drive_traces.append(drive_recorder.finish(state, PossessionTerminal.PUNT, points=0))\n            drives += 1\n            drive_recorder = DriveTraceRecorder(state)\n            initial_completed.add(offense.team_id)\n            if len(initial_completed) >= 2 and state.away_score != state.home_score:\n                break\n            continue\n\n        if event.play_type == PlayType.FIELD_GOAL:\n''',
    label="overtime punt terminal",
)
loop = replace_once(
    loop,
    '''            state = advance_game_clock(state, event.elapsed_seconds)\n            if fg.made:\n                state = _add_score(state, 3, away_team_id=away.team_id)\n                initial_completed.add(offense.team_id)\n                if sudden_death or (\n                    len(initial_completed) >= 2\n                    and state.away_score != state.home_score\n                ):\n                    break\n                if overtime_complete(state):\n                    break\n                state = _kickoff(state, rng, special)\n                drives += 1\n            else:\n                state = missed_field_goal_transition(state, elapsed_seconds=0)\n                drives += 1\n                initial_completed.add(offense.team_id)\n                if (\n                    len(initial_completed) >= 2\n                    and state.away_score != state.home_score\n                ):\n                    break\n            continue\n''',
    '''            state = advance_game_clock(state, event.elapsed_seconds)\n            if fg.made:\n                state = _add_score(state, 3, away_team_id=away.team_id)\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.FIELD_GOAL)\n                )\n                drive_recorder = DriveTraceRecorder(state)\n                initial_completed.add(offense.team_id)\n                if sudden_death or (\n                    len(initial_completed) >= 2\n                    and state.away_score != state.home_score\n                ):\n                    break\n                if overtime_complete(state):\n                    break\n                state = _kickoff(state, rng, special)\n                drives += 1\n                drive_recorder = DriveTraceRecorder(state)\n            else:\n                state = missed_field_goal_transition(state, elapsed_seconds=0)\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.MISSED_FIELD_GOAL, points=0)\n                )\n                drive_recorder = DriveTraceRecorder(state)\n                drives += 1\n                initial_completed.add(offense.team_id)\n                if (\n                    len(initial_completed) >= 2\n                    and state.away_score != state.home_score\n                ):\n                    break\n            continue\n''',
    label="overtime field goal terminal",
)
loop = replace_once(
    loop,
    '''            safeties += 1\n            initial_completed.add(offense.team_id)\n\n            # 2026 Rule 16 exception''',
    '''            safeties += 1\n            drive_traces.append(\n                drive_recorder.finish(state, PossessionTerminal.SAFETY, points=0)\n            )\n            drive_recorder = DriveTraceRecorder(state)\n            initial_completed.add(offense.team_id)\n\n            # 2026 Rule 16 exception''',
    label="overtime safety terminal",
)
loop = replace_once(
    loop,
    '''            state = _kickoff(state, rng, special)\n            drives += 1\n            continue\n\n        if event.turnover:\n''',
    '''            state = _kickoff(state, rng, special)\n            drives += 1\n            drive_recorder = DriveTraceRecorder(state)\n            continue\n\n        if event.turnover:\n''',
    label="overtime safety next drive",
)
loop = replace_once(
    loop,
    '''            state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)\n            drives += 1\n            initial_completed.add(offense.team_id)\n''',
    '''            state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)\n            drive_traces.append(\n                drive_recorder.finish(state, PossessionTerminal.TURNOVER, points=0)\n            )\n            drive_recorder = DriveTraceRecorder(state)\n            drives += 1\n            initial_completed.add(offense.team_id)\n''',
    label="overtime turnover terminal",
)
loop = replace_once(
    loop,
    '''            if game_ending_td:\n                initial_completed.add(offense.team_id)\n                overtime_touchdowns_without_try += 1\n                break\n''',
    '''            if game_ending_td:\n                drive_traces.append(\n                    drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN)\n                )\n                drive_recorder = DriveTraceRecorder(state)\n                initial_completed.add(offense.team_id)\n                overtime_touchdowns_without_try += 1\n                break\n''',
    label="overtime game-ending touchdown trace",
)
loop = replace_once(
    loop,
    '''            tries.append(trial)\n            state = _add_score(state, trial.points, away_team_id=away.team_id)\n            initial_completed.add(offense.team_id)\n''',
    '''            tries.append(trial)\n            state = _add_score(state, trial.points, away_team_id=away.team_id)\n            drive_traces.append(\n                drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN)\n            )\n            drive_recorder = DriveTraceRecorder(state)\n            initial_completed.add(offense.team_id)\n''',
    label="overtime touchdown trace",
)
loop = replace_once(
    loop,
    '''            state = _kickoff(state, rng, special)\n            drives += 1\n            continue\n\n        state = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)\n''',
    '''            state = _kickoff(state, rng, special)\n            drives += 1\n            drive_recorder = DriveTraceRecorder(state)\n            continue\n\n        state = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)\n''',
    label="overtime touchdown next drive",
)
loop = replace_once(
    loop,
    '''        if before.down == 4 and event.yards < before.distance:\n            state = turnover_on_downs(state)\n            drives += 1\n            initial_completed.add(offense.team_id)\n''',
    '''        if before.down == 4 and event.yards < before.distance:\n            state = turnover_on_downs(state)\n            drive_traces.append(\n                drive_recorder.finish(state, PossessionTerminal.TURNOVER_ON_DOWNS, points=0)\n            )\n            drive_recorder = DriveTraceRecorder(state)\n            drives += 1\n            initial_completed.add(offense.team_id)\n''',
    label="overtime downs terminal",
)
loop = replace_once(
    loop,
    '''    if state.seconds_remaining > 0 and len(plays) - len(regulation.plays) >= max_plays:\n        state = replace(state, seconds_remaining=0, quarter=5)\n\n    return GameResultV13(\n''',
    '''    if state.seconds_remaining > 0 and len(plays) - len(regulation.plays) >= max_plays:\n        state = replace(state, seconds_remaining=0, quarter=5)\n    if drive_recorder.has_activity:\n        drive_traces.append(drive_recorder.finish(state, PossessionTerminal.END_GAME))\n\n    return GameResultV13(\n''',
    label="overtime end terminal",
)
loop = replace_nth(
    loop,
    "        drives=drives,\n        defensive_stats=defensive_stats,\n",
    "        drives=drives,\n        drive_traces=tuple(drive_traces),\n        defensive_stats=defensive_stats,\n",
    occurrence=1,
    label="overtime result traces",
)
loop_path.write_text(loop)


# Integration guard: traces must remain deterministic and reconcile to the event-derived score.
test_path = Path("tests/test_game_loop_v13.py")
test = test_path.read_text()
test = replace_once(
    test,
    "from monster.sim.game_loop_v13 import simulate_regulation_game\n",
    "from monster.sim.football_state import PossessionTerminal\nfrom monster.sim.game_loop_v13 import simulate_regulation_game\n",
    label="integration test import",
)
test = replace_once(
    test,
    "    assert a.player_stats == b.player_stats\n",
    "    assert a.player_stats == b.player_stats\n    assert a.drive_traces == b.drive_traces\n",
    label="deterministic trace assertion",
)
test += '''\n\ndef test_drive_traces_reconcile_to_event_derived_scoring() -> None:\n    result = simulate_regulation_game(_team("away"), _team("home"), seed=66)\n    assert result.drive_traces\n    traced_touchdowns = sum(\n        trace.terminal == PossessionTerminal.TOUCHDOWN for trace in result.drive_traces\n    )\n    event_touchdowns = sum(play.touchdown for play in result.plays)\n    assert traced_touchdowns == event_touchdowns\n    traced_offensive_points = sum(trace.points for trace in result.drive_traces)\n    scoreboard_points = result.final_state.away_score + result.final_state.home_score\n    assert traced_offensive_points + 2 * result.safeties == scoreboard_points\n    assert all(trace.offense_team_id != trace.defense_team_id for trace in result.drive_traces)\n'''
test_path.write_text(test)
