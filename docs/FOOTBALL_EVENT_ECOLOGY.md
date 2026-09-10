# Monster Football Event Ecology

## Center

Monster is an independent, market-blind simulator of NFL reality. The core simulation unit is not "run/pass -> yards". It is a snap whose football interactions create an event, whose enforcement and possession consequences update one shared game state.

Canonical direction:

conditions -> pre-snap state -> personnel/formation -> offensive intent -> defensive intent -> snap -> blocking/trench interaction -> ball decision -> player interaction -> physical outcome -> loose-ball/foul overlay -> enforcement -> clock/down/distance/field/score transition -> next state

Final score, player statistics, fantasy points and later DFS value remain downstream consequences.

## Why this expansion is required

A football play can move forward, move backward, go nowhere, be erased, be replayed, change possession, create a score, stop or run the clock, or produce multiple sequential possession/ball states. Treating football as a small set of terminal labels with a sampled yard value systematically hides the failure ecology that makes real offenses inefficient.

Negative outcomes are first-class football events, not noise around positive-yard distributions.

## Event ecology

### Pre-snap and no-snap states

Personnel substitution, formation, alignment, motion, shift, audible/check, defensive disguise, blitz presentation, timeout, delay, false start, encroachment/neutral-zone event, illegal formation, illegal motion/shift, dead-ball foul, aborted/no-play state.

### Snap and exchange

Normal snap, low/high/bad snap, shotgun exchange, handoff/mesh, RPO mesh, botched exchange, immediate loose ball, QB sneak exchange, intentional clock-management snap.

### Offensive intent families

Dropback pass, quick game, play action, screen, RPO pass, designed rollout, designed QB run, scramble after dropback, inside/outside zone, gap/power/counter, draw, sweep/end-around, sneak, kneel, spike, trick/lateral concept, punt, fake punt, field goal, fake field goal, try.

These are intent/mechanism families, not terminal outcomes.

### Trench interaction

Protection win/loss, free rusher, edge win, interior pressure, blitz pickup, stunt/game, run-fit win/loss, penetration, displacement, pull/block success, missed assignment. These interactions should govern pressure timing, yards before contact, tackle-for-loss opportunity and whether the designed concept survives long enough to reach its next branch.

### Dropback resolution

Clean pocket, pressure, hit-as-thrown, batted/tipped pass, sack, strip-sack, scramble, throwaway, intentional grounding, target attempt, interception-worthy throw, ordinary incompletion, breakup, drop, interception, completion behind LOS, short/intermediate/deep completion, contested catch, created reception, YAC, tackle immediately after catch, missed tackle, fumble after catch, lateral after reception, touchdown.

A completion may have negative, zero or positive net yards. Air yards and YAC must not be forced to be independently nonnegative when their combined physical result can be a loss.

### Rushing resolution

Backfield penetration, tackle for loss, no gain, short gain, successful conversion, clean lane, contact at/behind/ahead of LOS, broken tackle, multiple-contact sequence, cutback, bounce outside, QB keeper/scramble, out of bounds, fumble, recovery by either team, explosive gain, touchdown.

Negative rushing outcomes should emerge from penetration, blocking, runner decision and tackling rather than from a generic negative-yard coefficient.

### Loose-ball and possession ecology

Fumble recovered by offense, fumble recovered by defense, strip-sack, muff, interception, interception return, lateral, failed lateral, turnover at spot, return touchdown, touchback, safety, turnover on downs. A play can contain more than one possession state before becoming dead.

### Penalty and enforcement ecology

Pre-snap dead-ball, live-ball accepted, declined, offsetting, replay down, automatic first down, loss of down, half-distance, spot foul, foul during return/loose ball, score plus enforcement, score erased by offense foul, defensive foul extending a drive. Penalties should eventually be subtype- and context-aware rather than a single aggregate yard distribution.

### Special teams ecology

Kickoff touchback/landing-zone return/out of bounds/onside, fair catch, punt return, punt out of bounds, downed punt, punt touchback, blocked punt, muff, field-goal make/miss/block, PAT, two-point play, fake kick, return after blocked/failed try where rules allow.

### Clock and boundary ecology

In-bounds tackle, out of bounds, incomplete pass, first down timing where applicable, score, timeout, two-minute warning, runoff, spike, kneel, penalty timing, end of quarter/half/game, overtime possession logic. Clock should emerge from event type and boundary outcome rather than one generic elapsed-time draw.

## Mechanism jurisdiction

Player and team evidence must enter where it has a football mechanism. Examples: OL/pass-rush evidence -> pressure/penetration; QB mobility -> escape/scramble response; route/catching/coverage -> target separation and catchpoint; weight/power/tackling -> contact survival; speed/acceleration -> separation, pursuit and open-field ceiling; wingspan/height -> catchpoint radius; wind -> throw/kick flight; health -> availability and capability distribution.

No rich feature should directly add fantasy points or generic yardage if a narrower football interaction can own it.

## Current evidence gates

The current v1.3 drive audit says drive frequency, drive length and first-down survival are near historical, while credited yards/explosives and red-zone finishing are high. Goal-line yardage conservation removed a substantial measurement artifact without changing any scoreboard outcomes. The next audits therefore have two separate questions:

1. Which play family owns the surviving explosive/gain excess?
2. Does Monster underproduce negative/no-gain outcomes, and in which family/mechanism?

Only after those are measured should behavior change.

## Calibration hierarchy

Validate distributions at league -> team -> player -> situation -> matchup -> game levels. Historical averages are falsification/calibration constraints, not objectives to be blindly matched. Realistic football must still generate unusually efficient and unusually inefficient games.

## Promotion rule

A green software test is not evidence of football realism. A plausible league average is not evidence of correct causality. Each promoted mechanism should have definition-matched historical evidence, causal ownership, invariants/conservation tests, paired before/after simulation evidence, and no upstream market leakage.
