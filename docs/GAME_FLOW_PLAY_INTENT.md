# Monster Game Flow and Play Intent

## Center

Monster should not choose a play from a static run/pass probability and then sample yards. A snap begins inside a strategic game state. That state creates an offensive problem. A coach/team chooses an intent and concept to solve that problem. Eleven offensive players and eleven defenders then interact, producing the physical football event.

Canonical chain:

conditions + roster availability + game state + drive history + coach/team identity + opponent identity -> strategic objective -> play-family intent -> concept intent -> personnel/formation/alignment -> matchup interactions -> execution/disruption -> event resolution -> enforcement -> next game state

Scores, player statistics, fantasy output, and later DFS value remain downstream consequences.

## Three different layers that must not be collapsed

### 1. Game Flow / Strategic Objective

The question is: **what does the offense need from this snap?**

Possible strategic objectives include:

- establish a normal series / stay on schedule
- gain enough to create a manageable next down
- convert the line to gain now
- create an explosive play
- reach field-goal range
- score before the half/game expires
- stop the clock
- drain the clock
- protect field position / avoid catastrophic loss
- force the defense to defend width, depth, run, or QB mobility
- exploit a known defensive weakness
- punish pressure/blitz
- manipulate a tendency established earlier in the game
- set up a future concept with play action, motion, constraint, or tendency breaking
- surrender some conversion probability to improve punt/field-position outcome

The objective is latent strategy. It is not itself a run or pass.

### 2. Play Intent / Tactical Call

The question is: **how does this offense try to solve the strategic problem?**

Top-level call families:

- designed run
- dropback pass
- play-action pass
- screen
- RPO
- designed QB run
- rollout / movement pass
- sneak
- draw
- trick/lateral concept
- spike/kneel
- punt / field goal / fake / try

Sub-intent should then describe the actual concept shape instead of a generic yard expectation.

Passing examples:

- behind-LOS screen / swing
- quick game
- sticks / conversion concept
- intermediate timing concept
- shot play
- bomb / Hail Mary
- max-protect vertical
- rollout / flood
- checkdown-access concept
- red-zone spacing/fade/pick-style family where evidence supports it

Rushing examples:

- inside zone
- outside zone
- power/gap
- counter
- draw
- sweep/toss/end-around
- short-yardage plunge
- QB sneak
- designed QB keeper/read
- constraint run against light box or pass expectation

Intent families are causal mechanisms, not terminal outcomes.

### 3. Play Resolution

The question is: **what actually happens once the ball is snapped?**

Resolution belongs to blocking, rush/coverage, QB decision, route execution, ball placement, catchpoint, penetration, contact, pursuit, loose-ball state, penalty state, boundary and clock behavior.

A perfectly logical call can fail. A questionable call can succeed. That variance is essential. Game-flow intelligence chooses the attempt; player interaction determines the outcome.

## State variables that should influence Game Flow

### Down and distance

Down/distance is one of the strongest strategic constraints but should not be reduced to one pass-probability table.

Examples:

- 1st-and-10: broadest call menu; establish tendency, run/pass balance, play action, scripted concepts, exploratory calls.
- 2nd-and-2: unusually rich freedom. The offense can convert cheaply, take a shot because 3rd-and-short is tolerable, use play action, or exploit the defense overplaying the run.
- 2nd-and-long: offense may try to recover schedule with efficient pass/screen/run, or remain aggressive depending on coach/team.
- 3rd-and-1/2: conversion dominates, but sneak, power, quick pass, play action, QB run and tendency breakers differ by personnel and coach.
- 3rd-and-medium: route-to-sticks structure, pressure response and QB mobility become important.
- 3rd-and-10: conversion urgency strongly increases dropback intent, target depth, route structure and pressure exposure.
- 3rd-and-18: not equivalent to ordinary 3rd-and-long. Field position, expected punt value, score/time and fourth-down status may justify screen/draw/checkdown instead of forcing a low-probability deep throw.
- 4th down: decision to punt/kick/go precedes concept selection; if going, required gain and game leverage should sharply constrain intent.

Exact yard-to-go should ultimately remain available rather than being permanently collapsed into short/medium/long buckets.

## Time and score

Clock and score change the value of yards, possession, incompletion, boundary outcomes and risk.

Important modes include:

- normal/open game
- early scripted drive
- end-of-first-half possession
- four-minute offense protecting a lead
- two-minute/hurry-up offense
- one-score comeback
- multi-score desperation
- goal-to-go clock tradeoff
- final-play Hail Mary/lateral desperation
- overtime possession logic

The same 2nd-and-5 should produce different strategy with 11:40 in Q1 tied, 1:12 in Q2 with two timeouts, 4:30 in Q4 up 10, or 0:18 in Q4 down 6.

## Field position and scoring horizon

Game Flow should know:

- own backed-up territory
- normal open field
- midfield transition
- opponent territory
- fringe field-goal range
- reliable field-goal range
- high red zone
- low red zone / goal-to-go
- safety-risk territory

Field position changes acceptable risk and the value of conversion, explosive intent, sacks, penalties, punts, and incomplete passes.

## Coach and team identity

League context supplies a prior. Team/coach identity supplies the deviation.

Potential behavioral traits, learned only where sample/evidence permits:

- early-down pass tendency
- first-down run/pass tendency
- second-and-short shot tendency
- short-yardage run/QB-sneak tendency
- third-and-long aggression versus surrender tendency
- fourth-down aggression
- play-action tendency by situation
- screen tendency versus pressure/long distance
- run concept family mix
- pass-depth mix
- no-huddle / tempo tendency
- clock-drain tendency with lead
- timeout usage
- red-zone run/pass/concept tendency
- tendency-breaking / sequential dependence

Coach identity should not become an arbitrary rating bonus. It should alter decision distributions at the decisions the coach plausibly owns.

## Opponent and 11v11 response

Intent should also react to the opponent, not only the scoreboard.

Examples:

- strong pass rush -> quick game/screens/max protection/movement passes may gain share
- light box -> run/RPO opportunity
- weak run front -> run aggression may persist
- poor tackling -> screens/YAC concepts may become more attractive
- vulnerable deep coverage -> shot-play opportunity
- blitz tendency -> hot/quick/screen responses
- elite coverage + weak run defense -> altered call mix

The defensive call/alignment should then interact with the offensive call after both intents are selected. The simulator should not let the offense know the exact random defensive outcome in advance.

## Personnel, health, substitutions and fatigue

A strategic state is incomplete without who is actually available.

Examples:

- backup QB may shrink deep/complex pass menu
- injured OL may change protection, screen, quick-game and run direction preferences
- unavailable power back can change short-yardage call family
- missing deep threat can reduce bomb intent
- mobile QB health can reduce designed-run/scramble willingness
- tired defensive front can increase late-game run effectiveness/opportunity
- personnel packages should change which concepts are available and who participates

Availability and capability should remain separate variables.

## Weather and environment

Weather should modify the decisions it plausibly affects rather than applying a global scoring penalty.

Examples:

- strong wind -> lower deep/bomb and long-field-goal preference; possibly more run/quick-game intent
- rain/wet field -> ball-security, footing, route/cut, kick and exchange uncertainty
- extreme heat/cold -> fatigue/handling effects where supported
- dome -> nullify wind/precipitation effects

## Sequence and within-game learning

Game Flow should eventually remember what has happened earlier in the game.

Useful sequence state may include:

- previous play family and outcome
- consecutive run/pass count
- recent pressure/blitz exposure
- recent explosive success/failure
- run-front success by lane/concept
- defensive response to motion/play action
- drive number and opening-script phase
- tendency shown versus tendency breaker

This must be bounded to avoid storytelling. Historical evidence should determine whether sequential effects materially improve realism.

## Example Flowers

### 3rd-and-10, Q2, tied, midfield

Strategic objective: convert while avoiding catastrophic field-position loss.

Likely tactical menu: sticks/intermediate pass, layered route concept, QB mobility escape, screen against pressure, occasional draw depending on coach/opponent.

Resolution then depends on protection, pressure, coverage, target separation, QB decision and catchpoint.

### 2nd-and-2, Q1, tied

Strategic objective has optionality because failure still leaves manageable 3rd down.

Possible intents: efficient run for first down, play-action shot, deep shot, RPO, tendency breaker. Coach aggression, explosive personnel and defensive alignment determine mix.

### 3rd-and-18, Q3, tied, own 25

Strategic objective may become expected-possession management rather than literal 18-yard conversion at any cost.

Possible intents: screen, draw, checkdown-friendly concept, deep conversion attempt. Field position, opponent rush, score, time and coach aggression govern the mixture.

### Final snap of half, outside normal scoring range

Strategic objective: score immediately or create a legal last-play scoring sequence.

Intent may collapse toward Hail Mary / lateral desperation. Ordinary neutral pass-depth priors should have essentially no authority here.

## Proposed runtime architecture

### GameFlowState

Derived before every snap from FootballState plus contextual state:

- exact down
- exact yards to go
- yardline / goal-to-go
- quarter and seconds remaining
- score margin
- offense/defense timeouts
- expected remaining possession horizon when supportable
- drive number / drive phase
- clock mode
- fourth-down territory
- field-goal-range state
- roster/health availability
- weather/environment
- recent sequence memory

### StrategicObjective

A probability distribution over goals such as:

- stay_on_schedule
- convert_now
- create_explosive
- protect_field_position
- reach_scoring_range
- score_now
- stop_clock
- drain_clock
- setup_constraint

### PlayIntent

A probability distribution over tactical call families conditioned on GameFlowState + StrategicObjective + coach/team/opponent identity.

### ConceptIntent

Sub-family such as screen/quick/intermediate/deep/bomb or zone/power/counter/draw/sneak/QB keeper.

### Resolution

Existing and future 1v1-to-11v11 interaction kernels resolve the chosen concept into pressure, penetration, target/run path, catch/contact, loose-ball, penalty and boundary events.

## Evidence plan

Do not hardcode the examples above as rules unless football rules require them. Use historical play-by-play to learn and falsify the behavioral distributions.

First passive audit should preserve exact or fine-grained situation and measure:

1. designed-run vs dropback family by down, exact/fine distance, field zone, quarter/time and score state;
2. throw-depth distribution within those same states;
3. run direction/gap where provider data supports it;
4. shotgun/no-huddle and other observable formation/tempo markers;
5. team deviations from league context with hierarchical shrinkage for sparse cells;
6. special states: second-and-short, third-and-long/extreme-long, goal-to-go, backed up, two-minute, four-minute, end-of-half, desperation;
7. sequential/tendency effects only as a later experiment.

The current 12-cell neutral situational pass table remains a valid baseline layer, but it is not the final game-flow model.

## Initial 2025 game-flow audit evidence

The first passive audit observed 32,813 regular-season scrimmage decisions across 3,290 fine-grained context cells without changing simulator behavior.

League-level special states show that play-family intent is highly nonlinear:

- second-and-2-or-less: 31.5% dropback / 68.5% designed run;
- third-and-short: 35.9% dropback / 64.1% designed run;
- third-and-medium: 86.7% dropback;
- third-and-7-to-10: 94.1% dropback;
- third-and-11-to-17: 91.4% dropback;
- third-and-18-plus: 80.3% dropback / 19.7% designed run;
- end of first half: 80.4% dropback;
- late trailing: 85.9% dropback;
- four-minute lead: 14.4% dropback / 85.6% designed run;
- low goal-to-go: 43.4% dropback / 56.6% designed run.

This falsifies a simple monotonic "longer distance -> more passing" model. Extreme long-yardage states reintroduce screen/draw/field-position behavior rather than forcing literal conversion attempts.

Pass-depth intent also changes by situation. Ordinary throws were approximately 21.6% behind LOS, 33.8% at 0-5 yards, 12.4% at 6-9, 21.2% at 10-19, 9.5% at 20-39 and 1.5% at 40+. Third-and-7-to-10 shifted toward 10-19-yard intent, while third-and-18-plus shifted strongly back toward behind-LOS and very short throws. This is evidence that pass depth must be conditioned on strategic state rather than sampled from one global distribution.

Second-and-short was strongly run-heavy. Within its pass attempts, the 0-5-yard band was the largest observed depth family; the first audit did not show a universal deep-shot explosion. Any second-and-short shot tendency should therefore be learned as a team/coach/personnel/context deviation rather than hardcoded as a league rule.

Team variation is material. With reasonable minimum sample counts, second-and-short team dropback rates ranged roughly 7%-56%, third-and-short roughly 18%-54%, end-first-half roughly 61%-96%, low-goal-to-go roughly 26%-66%, and four-minute-lead behavior roughly 0%-31% dropback. Sparse extreme-long cells are even more variable. This supports hierarchical team/coach identity layered over league context.

The audit also showed run-location shifts: middle runs become more common on third-and-short and fourth-and-short than in ordinary states. This is an early observable proxy for run concept intent; richer concept classification can be added as data support improves.

## Governance

- Game Flow chooses intent; it does not determine success.
- Coach identity changes decision distributions, not generic efficiency.
- Player evidence changes available concepts and execution mechanisms, not arbitrary play-call bonuses.
- Weather acts only through plausible decision/execution jurisdictions.
- Market data remains excluded upstream.
- Historical frequencies are priors/falsification targets, not instructions to reproduce league averages blindly.
- Sparse coach/team/situation cells must shrink toward broader context rather than overfit.
- No new behavior is promoted until definition-matched historical audits and paired simulation evidence show the mechanism improves realism.

## Current implementation seam

The live v1.3 engine currently asks `situation_policy` for one contextual pass probability and then samples `PASS` versus `RUN`; pass depth is subsequently drawn from a single shared air-yard distribution and run lane is sampled separately. The next implementation should insert GameFlowState and PlayIntent between FootballState and those resolution branches rather than replacing the event engine wholesale.

Recommended progression:

1. compile a hierarchical historical Game Flow prior from the new fine-grained context audit;
2. create non-authoritative GameFlowState / StrategicObjective / PlayIntent data structures and deterministic classification tests;
3. introduce play-family policy first while keeping the existing run/pass resolution kernels unchanged;
4. paired simulation audit of play-family decisions and game anatomy;
5. add pass concept/depth intent conditioned on Game Flow;
6. add run concept/lane intent conditioned on Game Flow;
7. only then deepen defensive counter-call, personnel packages, sequence memory and coach-specific adaptation.

## Current hypothesis from the September 10 audits

The current v1.3 pass/run family decision layer is directionally useful but too coarse. The passing engine overproduces intermediate intent and underproduces behind-LOS and deep/bomb intent; designed-run and scramble tails are compressed. The next architecture should therefore introduce Game Flow and Play Intent before resolving pass depth or run anatomy, so future distribution fixes are conditioned on why the play was called rather than applied globally.
