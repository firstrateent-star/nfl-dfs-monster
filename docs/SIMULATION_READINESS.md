# Monster Simulation Readiness

## Center

The Monster is an independent, market-blind NFL football-reality simulator. Simulation is part of model development; it is not synonymous with promotion or freeze.

The canonical causal order remains:

`football evidence -> player/current state -> unit/system reality -> matchup/environment -> possessions -> plays -> scoring -> players -> blind freeze -> DFS later`

A development branch may run diagnostic simulations before every future feature is mature. What matters is that each active mechanism is coherent, attributable, market-blind, reproducible, and explicit about whether it is production, provisional, partial, or shadow.

The older Monster Football Reality v1 release remains a bounded comparison baseline. The v1.3 sequential event engine is a development replacement candidate until it earns its own promotion and freeze gates.

## Readiness tiers

### R0 — Engine integrity

R0 answers: **Can this code generate coherent football worlds at all?**

Required:
- CI and rule/state-machine tests are green.
- The sequential game engine terminates without impossible state transitions.
- Score, possession, down/distance, field-position and opportunity accounting conserve correctly.
- Seeded runs are reproducible.
- Market-blind guards are intact.

R0 does not imply that the generated football distribution is realistic.

### R1 — Diagnostic simulation ready

R1 answers: **Can we run simulations that are trustworthy enough to diagnose the next causal defect?**

Required on the exact current commit:
1. R0 passes.
2. All intended slate teams compile and attach to the current Game Flow / intent / resolution path.
3. Current personnel and environment snapshots compile without material identity errors.
4. Active production mechanisms have explicit provenance and jurisdiction.
5. Incomplete mechanisms are explicitly marked PARTIAL/SHADOW and have no hidden authority.
6. The required comparison/audit chain executes end-to-end for the mechanism under study.
7. Paired experiments use the same games, seeds, world counts and upstream evidence on control and treatment.
8. Manifests state code version, seed, world count, market-blind status and known limitations.

At R1, 250-world paired experiments are appropriate for causal diagnosis. They are not release evidence.

### R2 — Evaluation simulation ready

R2 answers: **Is the current v1.3 anatomy healthy enough to evaluate as a whole football model rather than one local mutation?**

Required:
- Drive Survival Ecology is no longer materially distorted by known implementation defects.
- Fourth-down decision behavior has passed its post-repair paired audit.
- Clock/possession anatomy has a definition-matched historical comparison, including terminal-conditioned tails.
- Scoring-state ecology is measured and no known red-zone/goal-to-go bug is being compensated elsewhere.
- Run negative-play and explosive-tail anatomy are within an evidence-supported range.
- Pressure-conditioned pass outcomes have been compared against historical football evidence.
- Skill-player availability is world-specific and stable.
- Known OL/defensive availability limitations are either repaired or explicitly bounded for the evaluation.
- Conservation and market-blind gates remain green.

At R2, run at least two disjoint 1,000-world universes before discussing freeze candidacy. Evaluate score distributions, possession anatomy and player opportunity jointly; do not judge only mean game totals.

### R3 — Freeze-candidate simulation ready

R3 answers: **Can this exact football engine reasonably compete to become the new canonical frozen model?**

Required:
- Drive, clock, scoring, run and pressure anatomy pass their defined evidence gates.
- Availability propagates through skill players, offensive line and defensive units in sampled worlds.
- Current roster/health/environment inputs are fresh for the target slate.
- Any defensive tactical intent, assignment-level matchup, coaching/system identity, special-teams refinement, travel/rest or other expansion with production authority has its own provenance, coverage and causal evidence.
- All remaining experimental layers have zero production authority and are explicitly classified as SHADOW.
- Multi-seed stability passes at evaluation scale.
- Probability-distribution validation is available for completed historical/held-out games where definition-matched v1.3 backtests are possible.
- Frozen inputs, code SHA, seeds and output artifact hashes are reproducible.

R3 still does not authorize DFS translation. It authorizes a candidate to face the canonical Football Reality Freeze Gate.

### R4 — Canonical football freeze

R4 answers: **Has this exact model state earned canonical upstream authority?**

Required:
- `src/monster/audit/freeze_gate.py` passes on the exact candidate.
- Market blindness is proven.
- Current-state freshness is proven.
- Required anatomy/audit gates are green.
- Conservation and multi-seed stability are green.
- Provenance is complete for every production-authority input.
- No shadow feature has non-zero production authority.
- Freeze manifest and artifact hashes are written and immutable for the release.

Only after R4 may frozen football worlds flow into DFS scoring/economics and Monster150.

## Existing promoted / reusable foundations

These are established organs and should be reused rather than rebuilt unless evidence demonstrates a defect.

### Market blindness
- Sportsbook totals, spreads, moneylines and props are forbidden upstream.
- DFS salary, ownership and external fantasy projections are forbidden upstream.
- Contaminated v0.6.2 remains historical audit/control only.

### Historical evidence and score anatomy
- Regular-season PBP scope is explicit.
- Canonical team aliases are normalized before aggregation.
- Offensive TD labels use passing/rushing TDs rather than generic TD events.
- Historical drive, score, pressure, play-gain and other anatomy audits already exist and should be extended in place when new definition-matched fields are needed.

### Team identity / continuity
- Continuity-conditioned memory is an established evidence bridge.
- Continuity controls confidence in inherited identity; it is not a scoring bonus.
- Richer named coaching/system identity remains shadow until separately justified.

### Sequential game state
- v1.3 now has explicit shared scoreboard, clock, possession, down/distance, field position, punts, field goals, failed fourth downs, overtime and drive traces.
- Therefore sequential game-state scoring is no longer a future architectural requirement; its realism is now an evaluation problem.

### Matchup and player reality
- Existing unit-interaction, player reality, participation, health, offensive-line, defensive-unit and opportunity systems should be extended rather than replaced.
- World-specific skill availability exists in shadow form.
- The remaining availability seam is propagation into OL/defensive unit composition rather than creation of another availability engine.

## Current v1.3 readiness assessment

### Already sufficient for R1 diagnostic simulation
- Sequential v1.3 engine exists and has run complete slate-world experiments.
- Stage 2 Game Flow is certified and should remain frozen during Stage 3 diagnosis.
- Stage 3 pass-depth/run-intent ecology is attached across the Week 1 slate.
- DriveTrace and definition-matched historical drive audits exist.
- Drive-survival, scoring-state, clock/possession, pass-depth, run-geometry and pressure evidence paths exist.
- Market-blind and conservation tests exist.
- Paired Stage 3 workflows exist.

Therefore Monster **should continue diagnostic simulation now**. Waiting for every freeze blocker before simulating would prevent the simulations from doing their intended job: locating which causal mechanisms need repair.

### Current blockers to R2 whole-model evaluation
- Confirm the repaired fourth-down policy in a fresh paired Stage 3 run.
- Complete terminal-conditioned clock/drive-tail evidence after the historical drive clock-field repair.
- Repair ordinary early-down/series survival through its owning play-resolution mechanisms; do not tune final score directly.
- Compare pressure-conditioned Stage 3 outcomes against historical pressure evidence.
- Reassess run second-level/explosive anatomy after the survival repair.
- Keep scoring-state conversion bounded; current evidence indicates access to scoring territory is the larger issue than red-zone conversion itself.

### Current blockers to R3 freeze candidacy
- Propagate sampled availability through OL and defensive unit composition rather than hardcoding every unit player active.
- Run disjoint evaluation-scale seeds and demonstrate stability.
- Complete v1.3 probability-distribution validation on definition-matched completed games where possible.
- Promote only those newer defensive/coaching/assignment/ST/environment layers that earn evidence; otherwise keep authority at zero.
- Refresh target-slate personnel/health/environment state immediately before candidate freeze.

## Simulation cadence

Use simulation scale according to the question being asked:

- **Unit tests / micro worlds:** software correctness and conservation.
- **250-world paired slate:** local causal mutation diagnosis using identical control/treatment seeds.
- **Two or more disjoint 1,000-world slate universes:** whole-model anatomy and stability at R2.
- **Larger frozen runs:** only after the model is a credible R3 candidate and the extra worlds answer a real uncertainty/calibration question.

More worlds cannot repair a biased mechanism. Increase world count only after anatomy is credible enough that Monte Carlo error, rather than structural model error, is the uncertainty being reduced.

## Anti-duplication rule

Before creating a new model organ, audit, ledger, sampler or workflow, first determine whether an existing component owns the responsibility.

Preferred order:

`reuse -> expose -> extend -> repair -> consolidate -> create only if genuinely missing`

New diagnostics should normally enrich an existing evidence family rather than becoming permanent parallel architectures.

## Launch philosophy

The development loop is:

`simulate -> observe anatomy -> classify mismatch -> identify causal owner -> make one bounded change -> rerun paired -> retain/revert -> expand scale only when justified`

The freeze loop is stricter:

`healthy anatomy -> fresh state -> replicated seeds -> probabilistic validation -> provenance -> freeze gate -> immutable football release -> DFS downstream`

This separation keeps simulation useful as a scientific instrument while preventing a merely runnable model from being mistaken for a canonical one.
