# Monster Football Event Ecology

## Center

Monster is an independent, market-blind simulator of NFL reality. The core simulation unit is not "run/pass -> yards". It is a snap whose football interactions create an event, whose enforcement and possession consequences update one shared game state.

Canonical direction:

conditions -> pre-snap state -> personnel/formation -> offensive intent -> defensive intent -> snap -> blocking/trench interaction -> ball decision -> player interaction -> physical outcome -> loose-ball/foul overlay -> enforcement -> clock/down/distance/field/score transition -> next state

Final score, player statistics, fantasy points and later DFS value remain downstream consequences.

## Why this expansion is required

A football play can move forward, move backward, go nowhere, be erased, be replayed, change possession, create a score, stop or run the clock, or produce multiple sequential possession/ball states. Treating football as a small set of terminal labels with a sampled yard value systematically hides the failure ecology that makes real offenses inefficient.

Negative outcomes are first-class football events, not noise around positive-yard distributions.

The desired simulator is therefore a generative football grammar. Intent, matchup, execution, disruption, ball state, enforcement and clock/field consequence combine to create observable plays. We should not maintain a giant menu of handcrafted terminal outcomes when a smaller number of causal layers can generate them.

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

## Certified observational evidence at the current v1.3 center

The definition-correct drive audit shows drive frequency, drive length and first-down survival near historical reality, while yards per drive, explosive production and red-zone touchdown conversion remain high. Goal-line yardage conservation removed impossible credited yardage without changing any scoreboard outcome, proving that measurement/physics bugs and football-calibration problems must be separated.

The play-family gain audit localizes the surviving explosive mismatch. Designed runs and scrambles are not over-explosive; they are drastically under-explosive. The excess comes from completed passes. Relative to 2025 regular-season history, Monster completed passes produce about 39% too many 15+ yard gains, while completed-pass air yards are about 49% high and YAC is about 12% low. The result is a synthetic middle-big passing distribution rather than a realistic long-tail distribution.

The negative/no-gain audit supports the missing-failure hypothesis but rejects a generic "add more negative yards" fix. Across all scrimmage plays Monster produces about 11% fewer negative gains and about 24% fewer nonpositive gains. Designed runs lose yardage nearly as often as history, but the severity is wrong: only about 0.60% of designed runs lose at least two yards versus about 5.20% historically, and the simulated conditional rushing loss is roughly -0.87 yards versus -2.27 historically. At the same time, 15+ designed runs are roughly 88% too rare. The run distribution is too narrow on both tails.

Completed passes expose a direct structural gap: Monster currently produces no negative-yard or zero-yard completions. Historical 2025 completed passes were negative about 3.09% of the time and zero-yard about 1.52% of the time. Passing failure shares are also low in the current sample: incompletions, sacks and interceptions all occur less often than in the historical scrimmage population.

The pass-depth audit identifies why completed air yards are inflated. Among throws with known air yards, the 2025 historical mix is approximately 17.8% behind the line of scrimmage, 34.7% at 0-5 yards, 14.4% at 6-9, 21.5% at 10-19, 9.9% at 20-39 and 1.65% at 40+. Monster's corresponding mix is approximately 9.1%, 19.4%, 23.4%, 42.6%, 5.6% and 0%. Monster therefore throws about half as often behind the line, nearly twice as often at intermediate 10-19 depth, substantially less often deep, and not at all in the 40+ bomb class in the audited 12,000 worlds. It also completes intermediate throws about 25% too often and 20-39 yard throws about 57% too often.

This evidence makes a single global air-yard mean adjustment inappropriate. The next passing mechanism should be a context-conditioned mixture of pass intentions/depth families whose completion, pressure, turnover and YAC behavior differs by branch. Likewise, the next rushing mechanism should widen both the destructive and breakaway tails through trench/contact branches rather than shifting mean yards.

## Current mechanism-certification questions

The next implementation work should answer these separately:

1. Passing intent/depth ecology: how should screen/behind-LOS, quick, short, intermediate and deep intentions be selected from situation, team identity and personnel?
2. Passing execution ecology: within each intention, how should protection, pressure, target separation, throw quality, catchpoint and YAC create completion/failure distributions?
3. Rushing tail ecology: how should trench loss, ordinary fit, line win, second-level contact and broken-tackle/open-field branches create realistic TFL severity and breakaway frequency at the same time?
4. Red-zone compression: after gain ecology is corrected, why does Monster still convert red-zone possessions to touchdowns too easily when space is compressed?
5. Rules/event overlays: how should penalties, loose balls, special teams, clock/boundary states and other non-simple terminal outcomes become event-native rather than aggregate modifiers?

These questions are ordered by causal dependency, not by which aggregate stat currently looks worst.

## Calibration hierarchy

Validate distributions at league -> team -> player -> situation -> matchup -> game levels. Historical averages are falsification/calibration constraints, not objectives to be blindly matched. Realistic football must still generate unusually efficient and unusually inefficient games.

A useful calibration object is not only a mean. For each mechanism, inspect event share, negative/zero mass, ordinary-success mass, upper/lower tails, conditional severity, context splits, team/player dispersion and cross-variable conservation.

## Promotion rule

A green software test is not evidence of football realism. A plausible league average is not evidence of correct causality. Each promoted mechanism should have definition-matched historical evidence, causal ownership, invariants/conservation tests, paired before/after simulation evidence, and no upstream market leakage.
