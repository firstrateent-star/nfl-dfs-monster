# MONSTER STATE

Last updated: 2026-09-10
Active development branch: `feature/football-state-v13-clock`

## READ FIRST

The governing specification is `MONSTER_CONSTITUTION.md`.
The implementation-completeness ledger is `docs/FEATURE_REALITY_AUDIT.md`.
The development/evaluation/freeze thresholds are `docs/SIMULATION_READINESS.md`.
The older machine-readable v1 release manifest remains `docs/releases/monster_football_reality_v1_2026_week1.json`.

## Canonical classification

Three model states must not be conflated:

1. **Monster Beta22 v0.6.3** — frozen Statistical/Structural Baseline retained for comparison.
2. **Monster Football Reality v1** — promoted bounded architecture/release from 2026-09-09. Its promotion evidence remains valid for that architecture, but does not prove the newer sequential v1.3 event engine is calibrated.
3. **Monster v1.3 sequential event engine** — current DEVELOPMENT REPLACEMENT CANDIDATE. It is diagnostic-simulation ready, not canonically frozen.

The present work must not silently fall back to v1 merely because older documentation says “promoted,” and it must not call v1.3 promoted merely because its tests or simulations run.

## Mission

Monster is first an independent, market-blind NFL football-reality simulator. Game scores and player outcomes emerge from coherent finite football worlds; DFS scoring/economics remain downstream of the blind football freeze.

Canonical path:

`NFL evidence -> human/player reality -> football ability -> current player state -> unit reality -> system reality -> opponent interactions -> environment -> possessions -> plays -> scoring -> players -> blind freeze -> DFS -> actual results -> calibration`

## Current v1.3 build center

The current engineering center is vertical causal depth through:

`play resolution -> series survival -> drive survival -> scoring access -> possession/clock ecology -> score distribution`

Stage 2 Game Flow is treated as certified and should not be casually retuned while Stage 3 resolution/drive ecology is being diagnosed.

### Strong / reusable current organs

- Explicit sequential scoreboard, game clock, possession, down/distance and field position.
- Punts, field goals, failed fourth downs, turnovers, touchdowns and regular-season overtime.
- Event-derived player statistics and conservation checks.
- Certified Game Flow and hierarchical play-intent evidence.
- Stage 3 pass-depth and run-geometry intent/resolution path.
- Definition-safe `DriveTrace` plus historical drive reconstruction.
- Drive-survival, scoring-state and clock/possession audit families.
- Historical pressure evidence and Stage 3 pressure-conditioned comparison path.
- Player physical/Madden/current-state identity bridges and existing unit compilation.
- World-specific skill-player availability in shadow form.
- Market-blind guards and paired same-seed experiment workflows.
- Probabilistic score evaluation utilities.
- Canonical football freeze gate.

### Current causal diagnosis

Latest Stage 3 evidence localizes the major scoring mismatch primarily to possession survival rather than play intent:

- Third-and-long creation is approximately historical.
- Series conversion is too low.
- Three-and-outs are too high.
- Third-down conversion is too low.
- Early-down 5+ yard gains are too low.
- Fourth-down offensive aggression was materially too conservative and has received a bounded evidence-based repair that still requires fresh paired confirmation.
- Red-zone access is too low, while conversion once in scoring territory is not the primary deficit and has recently been somewhat high.
- Drive count is high and drive-duration tails are too compressed even though median duration is close to historical football.
- Run destructive outcomes are broadly plausible, while second-level/open-field explosive tails remain an expansion target.
- Pressure response is directionally plausible but must pass the definition-matched historical comparison before further tuning.

Do not solve these by tuning final score or moving toward sportsbook numbers.

## Readiness status

### R0 — Engine integrity
**Functionally established, subject to exact-commit CI.**

The v1.3 engine has already completed full slate-world simulations with seeded reproducibility and event conservation.

### R1 — Diagnostic simulation
**READY / ACTIVE DEVELOPMENT MODE.**

Use 250-world paired same-seed simulations to diagnose one bounded causal mutation at a time. Diagnostic simulation is part of building Monster and should continue now.

### R2 — Whole-model evaluation
**NOT YET CLEARED.**

Remaining gates:
- fresh paired confirmation of repaired fourth-down policy;
- completed terminal-conditioned clock/drive-tail comparison;
- evidence-owned repair of ordinary early-down/series survival;
- completed pressure-conditioned historical comparison;
- reassessment of run second-level/explosive anatomy after survival repair;
- explicit treatment/bound for current availability limitations;
- market-blind and conservation gates green on the exact candidate.

After these pass, run at least two disjoint 1,000-world slate universes and evaluate score distributions, possession anatomy and player opportunity together.

### R3 — Freeze candidate
**BLOCKED.**

Additional requirements beyond R2:
- sampled availability propagates through skill players, offensive line and defensive units;
- target-slate roster/health/environment state is freshly versioned;
- all nonzero-authority defensive/coaching/assignment/ST/environment expansions have evidence/provenance;
- unpromoted layers remain SHADOW with zero authority;
- multi-seed stability passes;
- v1.3 probabilistic validation exists on definition-matched completed/held-out games where feasible;
- exact inputs/code SHA/seeds/artifact hashes are reproducible.

### R4 — Canonical football freeze
**BLOCKED BY R2/R3.**

Only an R3 candidate may face `src/monster/audit/freeze_gate.py`. Only after R4 may football worlds feed the canonical DFS/Monster150 pipeline.

## Current personnel / availability seam

Do not create another availability architecture.

`src/monster/sim/availability_world.py` already owns world-level skill-player active/inactive sampling and role redistribution.

The remaining structural defect is that unit compilation currently treats OL/defensive/ST players as certainly active. The correct repair is to propagate the existing sampled availability concept through the existing unit-player inputs and recompute the existing unit effects for active personnel. This should alter protection, run blocking, pass rush, coverage and run defense causally rather than creating separate injury engines.

## Current environment seam

The older v1 architecture has certified/bounded environment mechanisms. The v1.3 development runner must separately prove that the current target-slate environment snapshot is actually attached to the sequential engine before R3. Do not inherit ACTIVE status from old v1 documentation without tracing the v1.3 causal path.

## DFS boundary cleanup

The v1.3 development runner currently has legacy convenience output that calculates FanDuel scoring after football events. Those values do not feed back into football, but canonical development/evaluation artifacts should become football-only so the repository matches the Constitution literally. DFS translation belongs after the football freeze.

## Anti-duplication rule

Before creating a new model organ, audit, ledger, sampler or workflow:

`reuse -> expose -> extend -> repair -> consolidate -> create only if genuinely missing`

Examples:
- extend `DriveTrace`; do not create a second drive ledger;
- extend Clock & Possession Ecology; do not proliferate separate clock engines;
- extend historical pressure evidence; do not create a competing pressure truth source;
- extend `AvailabilityWorld` into units; do not create OL/defensive injury engines;
- extend the existing freeze gate; do not create a second canonical freeze controller.

## Immediate sequence

1. Complete the fresh fourth-down paired Stage 3 gate.
2. Complete terminal-conditioned clock and pressure diagnostics using the repaired historical drive trace.
3. Use the existing Drive Survival evidence to identify the exact early-down/run/pass resolution owner and make one bounded mutation.
4. Rerun paired; retain or revert from anatomy, not score preference.
5. Reassess run second-level/explosive ecology.
6. Propagate world availability into OL/defensive unit composition through existing unit machinery.
7. Verify v1.3 environment/current-state attachment explicitly.
8. Reassess all R2 gates.
9. If R2 clears, run two disjoint 1,000-world whole-model evaluation universes.
10. Only after R2 evidence, work through R3 freeze-candidate blockers.

## Simulation law

Simulation scale follows the scientific question:

- unit/micro tests -> correctness;
- 250-world paired slate -> local causal diagnosis;
- disjoint 1,000-world slates -> whole-model evaluation/stability;
- larger frozen runs -> only after structural model error is small enough that Monte Carlo uncertainty is worth reducing.

More worlds do not repair a biased football mechanism.

## Constitutional boundary

Sportsbook totals/spreads/moneylines, DFS salary, ownership and optimizer outputs remain prohibited upstream. Market disagreement may be inspected only after a football freeze and is never an upstream correction target.
