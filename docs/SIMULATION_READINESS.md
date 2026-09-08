# Monster Simulation Readiness

## Center

The Monster is an independent, market-blind NFL slate simulator. Its production order is:

`football data -> frozen slate reality -> game/team score worlds -> player opportunity/stat worlds -> validation -> DFS translation later`

A mechanism is not required to be perfect before launch. It is required to be either (a) promoted by evidence, (b) explicitly bounded as a provisional input, or (c) deferred so it cannot masquerade as production authority.

## Promoted / production-authority layers

### Market blindness
- Sportsbook totals, spreads, moneylines and props are forbidden upstream.
- DFS salary, ownership and external fantasy projections are forbidden upstream.
- Contaminated v0.6.2 remains historical audit/control only.

### Historical score anatomy
- Regular-season PBP scope is explicit.
- Canonical team aliases are normalized before aggregation.
- Offensive TD labels use passing/rushing TDs rather than generic TD events.
- Possession calibration no longer double-applies pace.

### Offseason team identity
- Continuity-conditioned memory is the promoted bridge.
- It passed year-forward OOS validation across 2022-2025.
- Continuity controls confidence in inherited identity; it is not a scoring bonus.

### Shared game clock / finite opportunity
- Teams share one possession universe rather than independent possession lotteries.
- Teams share one finite play budget.
- Opportunity mean and tails are historically realistic.
- Shared-clock engine passed scoring noninferiority OOS.

### Matchup interaction kernel
The current production kernel expresses opponent-specific collisions through:
- pass protection vs pass rush + coverage,
- run blocking vs run defense,
- pass disruption,
- run efficiency,
- drive quality / turnover / scoring propagation.

A controlled 50,000-world elasticity audit passed all 9/9 monotonic direction checks. This establishes mechanical responsiveness, not historical injury-value calibration.

### Rushing opportunity anatomy
- Finite core rushing group plus incidental tail.
- Latent A/B/C hierarchy.
- Realized-world leader and rank anatomy structurally validated.

### Receiving opportunity anatomy
- Separate receiving hierarchy and eligibility state.
- Rank geometry, middle tier, peripheral tail and route/read eligibility validated with held-out and rolling OOS checks.

### Current health interface
Production currently carries separate availability/effectiveness/role inputs and is required to remain score-bounded.
Health is intentionally shallow for launch because live provider coverage is incomplete. It is not allowed to claim full replacement-chain realism yet.

## Measured but deferred

### Full sequential game-state scoring
Historical 2022-2025 PBP strongly confirms score-state behavior. In the final five minutes, teams trailing 7-13 points passed dramatically more often than teams leading 7-13, with the direction stable every season.

Production currently allows player opportunity to respond to score pressure, but the core scoring engine remains aggregate-drive rather than a drive-by-drive scoreboard state machine.

Reason for deferral: do not destabilize an OOS-tested score engine immediately before the first slate launch. Sequential scoring becomes a post-launch structural candidate and must earn its own OOS noninferiority/predictive gate.

### Deep health replacement realities
Deferred until better live health data are available. Future design:
`P(active) -> effectiveness|active -> sampled reality -> replacement chain -> rebuilt unit -> matchup -> scoring`.

### Field position / special teams state machine
Drive-start field position, punts/returns, failed fourth downs and detailed kicking range are not yet explicit sequential states.

### Richer coaching/system transitions
Current policy and continuity mechanisms capture part of this. Coordinator/HC transitions, personnel-package philosophy and opponent-specific adjustment behavior remain future work.

### Individual micro-matchups
Examples: LT vs specific EDGE, WR route archetype vs specific CB, TE vs LB/S, QB-specific pressure response. Current launch authority is at the unit-interaction level.

### Weather refinement
Environment hooks exist; richer empirically validated wind/rain/temperature/surface mechanisms remain future work.

### Astrology
Shadow experiment only. Production weight remains zero unless it earns independent OOS evidence.

## Launch blockers

The Monster is ready to begin true slate simulation when all of the following are true on the current production commit:

1. Market-blind production seal is intact.
2. Continuity-conditioned current team policy compiles.
3. Shared-clock / finite-play engine is active.
4. Player opportunity conservation/anatomy regressions remain green.
5. Current health inputs pass finite/bounded production-safety checks.
6. Full 60,000-world x 12-game rehearsal completes without technical failure.
7. QB/current-player identity audit shows no material roster-role ingestion error.
8. CI is green.
9. Output manifests explicitly state known data limitations rather than hiding them.

## Launch philosophy

The first production slate is a model version, not the end of model development.

Once this gate is green, the highest-value next activity is to simulate games, freeze predictions, observe real outcomes, score calibration/error, and use those errors to decide which deferred branch deserves the next Flower cycle. Real Week 1 evidence should outrank speculative pre-launch complexity.
