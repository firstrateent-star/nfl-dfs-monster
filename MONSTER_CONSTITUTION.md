# MONSTER CONSTITUTION

Status: GOVERNING SPECIFICATION
Established: 2026-09-08

## Mission
Monster is first an independent NFL football-reality simulator, not a DFS projection model or optimizer.

Its job is to reconstruct football reality as richly and causally as practical, simulate coherent Sundays, and allow game scores and player outcomes to emerge. DFS scoring, salary, ownership and portfolio construction are strictly downstream.

Canonical path:
NFL evidence -> human/player reality -> football ability -> current player state -> unit reality -> system reality -> opponent interactions -> environment -> possessions -> plays -> scoring -> players -> blind freeze -> DFS -> actual results -> calibration.

## Non-negotiable original scope
A model may not be called Full Monster / Football Reality v1 until each category below is either ACTIVE with evidence, deliberately SHADOW, or explicitly documented as unavailable with a tested fallback. Merely ingesting a field does not count as using it.

### 1. Historical football performance
Team and player historical statistics; offense and defense; scoring, yards, efficiency, usage, TD composition, situational performance and opponent-adjusted evidence where justified.

### 2. Human / physical reality
Player age, NFL experience, height, weight, body/size measures, wingspan/length where available and relevant, athletic testing and traits such as speed, acceleration, agility, strength and explosiveness.

Physical data must affect plausible football mechanisms rather than act as arbitrary fantasy-point bonuses.

### 3. Madden / scouting-style ability evidence
Madden attributes are an intended input family and MUST NOT silently disappear from Full Monster. Relevant attributes can include speed, acceleration, agility, strength, awareness, throw power/accuracy, carrying, catching, route running, release, break tackle, blocking, tackling, coverage and pass-rush traits.

Madden is evidence, not truth. It should be blended/validated against empirical football evidence and should influence causal matchup mechanisms. Example: receiver release/route/speed interacting with defender coverage/speed; QB pressure traits interacting with pass rush/protection; RB burst/strength/break-tackle interacting with blocking/front/tackling.

### 4. Current circumstance
Injuries, availability, effectiveness-if-active, recovery uncertainty, depth chart, starter status, workload, recent role/usage and personnel changes.

### 5. Unit reality
QB room, backfield, receiving corps, offensive line, defensive front, linebackers, secondary and special teams. Unit quality must emerge from relevant personnel plus empirical unit evidence rather than team-name priors alone.

### 6. Coaching / system / continuity
Coaching, scheme, coordinator/system changes, continuity, pace, personnel tendencies, pass/rush tendencies and situational behavior.

### 7. Matchup interaction
Opponent effects must be mechanistic where feasible: OL vs rush/front, receiver vs coverage, QB vs pressure/coverage, rushing traits/blocking vs front/tackling, etc. Independent player projections pasted into a game are insufficient.

### 8. Environment
Home field plus relevant weather, wind, temperature, precipitation, venue, surface, travel/rest or other conditions when evidence supports an effect.

### 9. Finite correlated football worlds
Every world must respect finite possessions/plays/opportunities and preserve appropriate within-player, team, game and opponent correlations. Targets, rushes, yards, TDs and red-zone opportunities cannot be independently manufactured without conservation.

### 10. Uncertainty
Persistent player/team identity must be separated from current circumstance. Expected-state means must be separated from world-level tails. Missing/uncertain information is represented as uncertainty, not silently converted to certainty.

### 11. Experimental / Shadow Monster
Astrology/birth-chart signals and other speculative features remain separately governed, shadow-only and zero-authority until prospective/out-of-sample evidence justifies promotion. They may never contaminate the baseline experiment silently.

## Causal implementation law
Features should alter football mechanisms, not directly award fantasy points merely because a rating is high.

Preferred chain:
traits + state + opponent + system + environment -> play probabilities/outcomes -> drives/score -> player statistics -> DFS scoring.

## Market-blind law
Sportsbook totals, spreads, moneylines, DFS salaries, ownership and optimizer outputs are prohibited upstream of the blind football freeze.

After freeze, market information may be revealed only for audit. Disagreement is a diagnostic coordinate, not a correction target.

Required comparison when available:
Frozen Monster -> market at reveal -> closing market -> actual NFL result.

## Baseline versus Full Monster
Monster Beta22 v0.6.3 and its Week 1 60K run are retained as a Statistical/Structural Baseline. They are NOT evidence that the full original Monster specification has been implemented.

No downstream OLR or lineup result generated from that baseline may be represented as Full Monster output.

## Feature Reality Audit
Every intended input family must be tracked as one of:
- ACTIVE: demonstrably changes simulated football worlds through a documented mechanism.
- PARTIAL: some intended information/mechanisms exist but scope is incomplete.
- INGESTED_BUT_INACTIVE: data exists but does not causally affect worlds.
- ABSENT: not implemented.
- SHADOW: intentionally isolated experimental input.

A passing unit test is not sufficient for ACTIVE status. Promotion requires provenance, coverage, causal path, conservation/interaction tests where applicable, and blind evidence showing the mechanism behaves in the intended direction without pathological effects.

## Promotion gate: Football Reality v1
The label `Monster Football Reality v1` is prohibited until:
1. the Feature Reality Audit covers the complete original scope;
2. Madden/scouting-style ability evidence is ACTIVE or an explicit evidence-backed substitute is approved;
3. physical/athletic/age/experience mechanisms are audited;
4. current personnel/health, units, coaching/system/continuity and environment are audited;
5. matchup mechanisms are audited;
6. finite opportunity/correlation gates pass;
7. multiple seeds/world counts show stability;
8. football output is frozen before market/DFS reveal;
9. a machine-readable manifest records every active/partial/shadow family and source snapshot.

Until then the implementation must identify itself as a baseline or development candidate.

## Strange-result law
A surprising result is classified by causal trace as SIGNAL, BUG, STATE MISS, MODEL MISS or UNCERTAINTY. It is never moved toward Vegas merely because it is surprising.

## Calibration law
Actual NFL results are used longitudinally to calibrate probabilistic mechanisms. One game is not a simplistic right/wrong grade. Track calibration, likelihood, distributional coverage and mechanism failures across slates.

## DFS boundary
Only after football worlds freeze may Monster apply FanDuel scoring, salary, OLR, ownership and portfolio optimization. The optimizer must never decide what football reality looks like.

## Anti-drift rule
Before any architecture freeze, release, Monster 150 generation or claim that a major layer is complete, compare implementation state against this file and the Feature Reality Audit. Missing original-scope categories must be surfaced explicitly.

If a future chat/session loses context, this repository specification overrides conversational shorthand. Read this file before continuing Monster architecture work.
