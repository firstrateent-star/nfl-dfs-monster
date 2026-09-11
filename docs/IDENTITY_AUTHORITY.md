# Monster Identity & Conditions Authority

## Purpose

Monster's football engine should first learn how NFL football behaves, then allow the specific humans, units, systems and conditions in a game to bend that league ecology.

The governing probability object remains:

`P(Game World | Football Evidence)`

No identity or condition feature may directly add points or fantasy production.

## Architecture

Evidence flows through four governed layers:

1. **Evidence source** — what was observed or supplied.
2. **Mechanism channel** — the football capability the evidence is allowed to inform.
3. **Interaction scale** — where that capability can matter.
4. **Event resolution** — the football event that changes the world state.

Example:

`Madden throw-under-pressure -> QB pressure-response channel -> QB vs rush/protection interaction -> sack/scramble/throw branch`

not:

`Madden rating -> +points`

## Current evidence families

### Human / physical

- height
- weight
- wingspan
- forty time
- age and career workload for uncertainty/durability only
- current availability
- effectiveness if active

### Madden / scouting-style proxy capability

Currently routed or available for routing:

- speed
- acceleration
- route running
- catching
- release
- carrying
- break tackle
- awareness
- throw power
- throw accuracy
- throw under pressure
- ball-carrier vision
- pass block
- run block
- pass rush
- coverage
- tackle
- kicking / return traits

Madden is proxy evidence, never ground truth and never market evidence.

### Historical football performance

- offensive/defensive EPA
- success rate
- explosive rate
- pressure / sack / hit evidence
- drive and scoring anatomy
- play intent and game-flow tendencies
- player opportunity / role evidence

### Current conditions

- game-day availability
- effectiveness if active
- continuity
- temperature
- wind
- precipitation
- dome / roof assumption
- surface is stored and traced but receives zero behavioral authority until a definition-safe mechanism is validated

## Player identity channels

`PlayerIdentityChannels` is the extension seam between raw evidence and the event engine.

Current channels:

- speed
- mobility
- QB execution
- route separation
- catchpoint
- rush creation
- runner power
- open-field ability
- ball security

A new metric should normally enrich one of these channels before a new channel is invented.

## Interaction scales

Monster's `InteractionRegistry` governs where evidence is allowed to matter:

- player
- 1v1
- small group
- unit
- 11v11
- game state

This lets the same source have different causal roles without granting unrestricted authority.

Examples:

- WR release: route separation in a 1v1 / small-group interaction
- CB coverage: route/catchpoint opposition in 1v1 and unit interactions
- OT pass block vs EDGE rush: pressure development in a 1v1 trench assignment
- five-man OL continuity: communication/protection at unit and 11v11 scale
- QB mobility vs contain: pressure-response / scramble branch
- RB vision + blocking geometry vs defensive front: second-level access

## Current v1.3 repair

The original v1.3 bridge compressed identity through generic `efficiency`, `explosive` and `turnover_security` values. It also did not give QB-specific Madden throwing attributes a direct route into Stage 3 pass resolution, and the v1.3 run context referenced an `offense_strength` field that the current policy compiler does not populate.

The identity-authority repair therefore:

1. preserves the existing Stage 3 league outcome priors;
2. compiles position-specific capability channels;
3. gives QB execution a bounded route into pass resolution and pressure response;
4. gives receiver capability a bounded route into target/catchpoint resolution;
5. gives runner capability a bounded route into run-contact / breakaway ecology;
6. restores a live market-blind team run context from existing EPA, success and explosive evidence rather than the unpopulated generic strength field;
7. leaves scoring event-derived;
8. keeps the pre-repair path available as a paired control until validation supports promotion.

## Environment authority

The repository already contains environment mechanisms. A v1.3 environment seam now resolves one shared game environment by venue/home team and applies it equally to both offenses.

Currently authorized behavioral inputs:

- temperature
- wind
- precipitation probability
- dome flag

Surface, roof nuance, humidity, altitude, travel and rest may be added to the evidence object immediately, but should remain trace-only until their football mechanism and historical calibration are defined.

## Future additions

Potential evidence can be added without redesigning Monster:

- Next Gen Stats / tracking speed and acceleration
- separation and route geometry
- time to throw
- pass-rush win rate
- pass-block win rate
- defender alignment
- box count / shell
- man/zone identification
- motion
- personnel packages
- individual matchup history
- hand size / arm length / vertical / broad jump
- GPS workload / fatigue
- rest and travel
- altitude / humidity / surface traction
- coaching/system identity
- assignment-level offensive and defensive intent

The rule is always the same:

`new evidence -> governed channel -> valid interaction scale -> football mechanism -> event -> world state`

## Promotion standard

More dispersion is not automatically better.

Identity authority can be promoted only if paired and out-of-sample evidence shows that it:

- increases sensitivity to real personnel differences;
- preserves or improves play/drive/scoring anatomy;
- produces monotonic counterfactual response to controlled talent/health changes;
- does not destabilize league scoring or possession ecology;
- remains market blind;
- does not double-count health, role, unit or historical team evidence;
- remains stable across seeds;
- leaves experimental layers at zero authority unless separately promoted.
