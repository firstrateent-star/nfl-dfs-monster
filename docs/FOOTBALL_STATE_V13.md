# Monster Football Reality v1.3 — Stateful Football Engine

Status: DEVELOPMENT BRANCH

## Center

The simulator's primary objective is not to predict a final score or a fantasy projection. It is to simulate a plausible NFL game such that score, team statistics, player statistics, and fantasy scoring emerge from the same finite football events.

## Governing causal chain

pre-game evidence
→ available personnel
→ game conditions
→ kickoff / initial possession
→ quarter + clock + score + field position + down + distance
→ personnel and play decision
→ offense/defense/player interaction
→ play event
→ yards / penalty / sack / turnover / score
→ updated football state
→ next snap / change of possession
→ four quarters / overtime
→ final score
→ player box score
→ downstream DFS scoring

## Non-negotiable invariants

1. The scoreboard is accounting over scoring events; it is never sampled independently.
2. Both teams consume the same finite game clock.
3. Every snap belongs to one possession and has a pre-snap football state.
4. Down, distance and field position update from the preceding play.
5. A turnover, punt, kickoff, failed fourth down or missed field goal determines the opponent's next field position.
6. Fourth-down decisions are football decisions conditioned on field position, distance, score and clock.
7. Pass/run/scramble/sack/penalty/turnover outcomes are distinct causal events rather than interchangeable yardage draws.
8. Player statistics are sums of player participation in simulated plays. A reception cannot exist without a target/completed pass; a receiving TD cannot exist without its scoring reception; a rushing TD cannot exist without its scoring carry.
9. Opportunity is conserved: snaps, attempts, targets, carries, receptions, touchdowns and turnovers cannot be independently inflated downstream.
10. Game situation changes future strategy. Score differential, time remaining, down/distance and field position must be able to change play selection and pace.
11. Physical, athletic, health, Madden/scouting, unit, coaching/system and environmental evidence may affect football mechanisms only through bounded documented paths. Missing evidence remains neutral rather than fabricated.
12. Market data, DFS salary, ownership and optimizer output have zero upstream authority.

## Development sequence

### v1.3A — possession continuity
Replace the v1.2 reset-to-own-25 possession model with stateful possession starts. Kickoffs, punts, turnovers, failed fourth downs and missed field goals hand a resulting field state to the opponent.

### v1.3B — game clock and quarter
Give each snap elapsed time, quarter/time remaining, halftime transition and end-of-game termination. Possession volume becomes a consequence of the finite clock rather than a preallocated drive count.

### v1.3C — decision policy
Condition pass/run, fourth-down attempts, field goals, punts, tempo and late-game behavior on football state. Historical football policy is evidence; sportsbook expectations are forbidden.

### v1.3D — richer play anatomy
Represent incomplete passes, sacks, QB scrambles, designed QB runs, interceptions, fumbles, penalties and special-teams transitions explicitly.

### v1.3E — player-on-play jurisdiction
Move player identity into the play itself: passer, intended receiver, runner and relevant matchup/unit context are selected before the result. Box-score allocation becomes aggregation of the event ledger rather than a separate statistical redistribution layer.

### v1.3F — calibration and promotion
Compare simulated football anatomy with historical NFL distributions: plays, drives, starting field position, down/distance, possession duration, pass/run by state, fourth-down choices, punts, field goals, turnovers, explosive plays, scoring anatomy and player opportunity distributions. Calibration may adjust football mechanisms from historical evidence but never toward sportsbook lines or DFS outcomes.

## Promotion rule

v1.3 cannot replace the promoted v1 architecture merely because its scores look plausible. Promotion requires causal invariants, historical football-anatomy calibration, multi-seed stability, source/evidence traceability, and blind football output frozen before any market or DFS reveal.
