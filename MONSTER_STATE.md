# MONSTER STATE

Last updated: 2026-09-08
Branch: `feature/slate-simulator-v1`

## READ FIRST
The governing specification is `MONSTER_CONSTITUTION.md`. The implementation-completeness ledger is `docs/FEATURE_REALITY_AUDIT.md`.

A scope-drift audit on 2026-09-08 found that the previously frozen football engine did not have sufficient evidence that all originally required Full-Monster feature families—most importantly Madden/scouting-style ratings and the complete physical/athletic/biology layer—were causally active.

Therefore the prior label **Monster Football v1 — ARCHITECTURE FROZEN** is RETRACTED as a Full-Monster completeness claim.

## Current canonical classification
**Monster Beta22 v0.6.3 — Statistical/Structural Baseline (FROZEN BASELINE, NOT FULL MONSTER)**

The baseline remains valuable and reproducible. Its football-only architecture passed its stated structural gates, but those gates certify the implemented baseline rather than the complete original Monster specification.

Freeze evidence: GitHub Actions run `34292058208`, commit `f33b6d693730443facce9a9e2f2f709c6c46976a`, seed `2026090816`, 60,000 worlds/game across 12 games = 720,000 simulated game-worlds.

## Mission
Monster is an independent NFL slate simulator. It must reconstruct football reality from historical performance, player physical/athletic/biological traits, Madden/scouting-style ability evidence, current personnel/health/roles, units, coaching/system/continuity, opponent matchup interactions and environment before sportsbook, salary, ownership or DFS optimization information is introduced.

Canonical path:
NFL evidence -> human/player reality -> football ability -> current player state -> unit reality -> system reality -> opponent interactions -> environment -> possessions -> plays -> scoring -> players -> blind freeze -> DFS -> actual-results calibration.

## Baseline capabilities retained
- Market-blind Beta22 game environment.
- Historical offense scoring/yards and opponent defense points/yards allowed.
- TD composition and home-field shift.
- Continuity-conditioned team identity.
- Shared finite game play supply.
- Receiving hierarchy and eligibility gates.
- Rushing anatomy / role structure.
- Health availability, effectiveness-if-active and uncertainty.
- Opportunity conservation.
- QB season-mean rushing calibration and finite QB reservoir.
- World-level rushing and target audits.
- Current personnel/OL/participation mechanisms already implemented.
- Correlated game/player worlds.
- Downstream FanDuel scoring, D/ST, exact lineup optimizer and OLR research infrastructure.

## What the baseline does NOT prove
It does not prove that the original Full-Monster scope is complete. In particular, Full Monster promotion is blocked until the Feature Reality Audit verifies and evidence-gates:
- Madden/scouting-style player ability ratings;
- height, weight, wingspan/length where relevant/available;
- athletic traits/testing such as speed, acceleration, agility, strength and explosiveness;
- age and NFL experience;
- complete unit/personnel interactions;
- coaching/scheme/tendency mechanisms;
- trait-level QB/receiver/RB/blocking/coverage/pass-rush matchup mechanisms;
- weather/venue/surface and other approved environmental effects.

These features must affect football mechanisms rather than receive arbitrary fantasy-point bonuses.

## 60K structural baseline evidence
Run `34292058208` completed successfully with market/contaminated references loaded=false, salary used=false and ownership used=false.

Broad structural Player Reality gate passed 20/20 and QB rushing gate passed. These remain valid evidence about the baseline implementation, not Full-Monster certification.

### Frozen baseline game means
- BAL@IND 23.58–23.77 (47.36)
- TB@CIN 23.83–23.09 (46.92)
- WAS@PHI 20.56–24.19 (44.74)
- NO@DET 19.93–24.68 (44.61)
- CHI@CAR 23.52–21.03 (44.55)
- BUF@HOU 23.22–21.08 (44.30)
- ATL@PIT 21.44–22.20 (43.64)
- ARI@LAC 19.58–22.97 (42.55)
- GB@MIN 21.39–20.77 (42.15)
- NYJ@TEN 20.66–21.41 (42.07)
- MIA@LV 21.83–18.98 (40.81)
- CLE@JAC 18.18–21.97 (40.15)

Artifact: `monster-week1-60k-final-blind`, artifact id `10081863532`.

## DFS/OLR status
FanDuel scoring, D/ST, salary identity matching and exact OLR machinery are retained as downstream research infrastructure. Existing OLR is conditional on the Statistical/Structural Baseline and MUST NOT be represented as Full-Monster OLR.

Monster 150 promotion is paused until Football Reality v1 passes the Constitution.

## Current development target
**Monster Football Reality v1 candidate**

Required sequence:
1. complete Feature Reality Audit;
2. establish canonical trait data/provenance;
3. implement missing physical/athletic/Madden/scouting/current-state/system/environment mechanisms;
4. counterfactual and ablation tests;
5. multiple blind seeds/world counts;
6. blind Full-Reality candidate freeze;
7. only then reveal market for diagnostic comparison;
8. rerun downstream FanDuel/OLR from promoted worlds;
9. build Monster 150 only after promotion.

## Constitutional boundary
Football reality remains independent of sportsbook lines, DFS salaries, ownership and optimizer feedback. Market disagreement is diagnostic, never an upstream correction target.

## Status
- Statistical/Structural Baseline: PASS / frozen
- Full original-scope audit: IN PROGRESS
- Madden/scouting ability layer: BLOCKER / unverified
- Physical/athletic/biology layer: AUDIT REQUIRED
- Current state/system/environment completeness: AUDIT REQUIRED
- Football Reality v1: NOT YET PROMOTED
- Existing FanDuel/OLR infrastructure: RETAINED DOWNSTREAM
- Monster 150: PAUSED pending Football Reality v1
