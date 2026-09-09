# MONSTER STATE

Last updated: 2026-09-08
Branch: `feature/slate-simulator-v1`

## What we are building
The Monster is an independent NFL slate simulator. Its primary job is to simulate coherent football worlds from football/context evidence before any sportsbook, salary, ownership, or DFS optimization information is introduced.

Canonical path:
NFL information -> personnel reality -> team reality -> game reality -> player reality -> simulated Sundays -> blind freeze -> DFS scoring -> salary -> player value -> ownership -> lineup value -> portfolio -> actual-results calibration.

## Primary objective
Predict distributions of NFL game and player outcomes by simulating finite, correlated worlds. DFS value is downstream of football reality; the optimizer must never decide what football reality looks like.

## FROZEN FOOTBALL ARCHITECTURE
**Monster Football v1 — Week 1 ARCHITECTURE FROZEN**

Freeze evidence: GitHub Actions run `34292058208`, commit `f33b6d693730443facce9a9e2f2f709c6c46976a`, seed `2026090816`, 60,000 worlds/game across 12 games = 720,000 simulated game-worlds.

The architecture is frozen before sportsbook, FanDuel salary, ownership, or optimizer information is introduced. Football-only current-state inputs (personnel, injuries/health, weather/conditions) may still be refreshed before slate lock, but each refresh must be versioned and may not use DFS/market information to alter football mechanisms.

## Retained architecture
- Monster Beta22 v0.6.3 market-blind Game Reality.
- Continuity-conditioned team identity.
- Shared finite game play supply.
- Receiving hierarchy and eligibility gates.
- Rushing anatomy / role structure.
- Health separated into availability, effectiveness-if-active, and uncertainty.
- Opportunity conservation.
- QB season-mean rushing calibration.
- Finite world-level QB rushing reservoir.
- Downstream rushing hierarchy preserves QB carry/TD reservoir.
- World-level rushing and target rank audits.
- Current personnel, OL, physical-mechanism and participation state.
- Evidence-gated promotion.

## Final 60K blind certification — PASS
Run `34292058208` completed successfully. Market/contaminated references loaded: false. Salary used in football simulation: false. Ownership used in football simulation: false.

### Broad Player Reality — PASS 20/20
- player rows: 514
- team rows: 24
- starter QB count: 24
- team plays min/median/max: 53.038 / 60.393 / 65.189
- pass attempts min/median/max: 26.118 / 30.760 / 36.009
- rush attempts min/median/max: 23.232 / 27.445 / 31.384
- targets min/median/max: 24.498 / 28.855 / 33.765
- yards/play min/median/max: 4.802 / 5.665 / 6.380
- target rank shares mean: 0.3316 / 0.2216 / 0.1579
- rush rank shares mean: 0.6310 / 0.2243 / 0.1016
- target earners/world mean: 7.212
- core rushers/world mean: 2.530
- starter QB pass attempts median/p90: 29.999 / 33.202
- starter QB passing yards median/p90: 219.893 / 238.010
- starter QB passing TD median/p90: 1.409 / 1.689

### QB Reality reconfirmation — PASS
- rush mean: 3.81953
- rush median: 3.50910
- rush p90: 6.20358
- rush p95: 7.23580
- rush max: 7.81245
- rushing-TD median: 0.13567
- rushing-TD p90: 0.41596
- rushing-TD max: 0.51717

### 60K game environment map
- BAL@IND: mean 23.58–23.77, total 47.36, P90 total 70, P(60+) 22.95%
- TB@CIN: 23.83–23.09, total 46.92, P90 70, P(60+) 22.06%
- WAS@PHI: 20.56–24.19, total 44.74, P90 67
- NO@DET: 19.93–24.68, total 44.61, P90 67
- CHI@CAR: 23.52–21.03, total 44.55, P90 67
- BUF@HOU: 23.22–21.08, total 44.30, P90 66
- ATL@PIT: 21.44–22.20, total 43.64, P90 65
- ARI@LAC: 19.58–22.97, total 42.55, P90 64
- GB@MIN: 21.39–20.77, total 42.15, P90 63
- NYJ@TEN: 20.66–21.41, total 42.07, P90 63
- MIA@LV: 21.83–18.98, total 40.81, P90 62
- CLE@JAC: 18.18–21.97, total 40.15, P90 61

60K distributions were stable relative to the prior 10K structural sample; no evidence-backed tail catastrophe was found that warrants changing football architecture before freeze.

Artifact: `monster-week1-60k-final-blind`, artifact id `10081863532`.

## Performance baseline
Run `34292058208` runner execution was about 6m10s. The 720,000-game-world simulation consumed ~339.8s (5m39.8s), approximately 2,119 game-worlds/sec and ~92% of runner time. Detailed timing is retained in `docs/PERFORMANCE_LEDGER.md`.

## Current-data policy
Provider injury rows were 0 on Sep 8; the Week 1 health snapshot used 13 explicit Sep 7 overrides. Official Sunday injury reporting had not yet populated when audited. This does not invalidate the architecture freeze. Personnel/health/weather must be refreshed from football-only evidence before slate lock and recorded as a new current-state snapshot.

Jacoby Brissett/Arizona and Tua Tagovailoa/Atlanta were previously suspected as mapping errors but current evidence supported those assignments at audit time. Reverify rather than changing surprising personnel by intuition.

## DFS transition
The blind football architecture is now frozen. Downstream work may begin without feeding results back upstream:
1. apply complete FanDuel scoring, including player-level turnover penalties once attribution is modeled;
2. join the FanDuel slate identity/salary file;
3. build multidimensional Monster Player Value distributions;
4. compute conditional-world/correlation evidence;
5. add ownership only downstream;
6. build and evaluate legal lineups/world portfolios;
7. eventually produce Monster 150.

## Constitutional boundary
Football reality must remain independent of sportsbook lines, DFS salaries, ownership, or optimizer feedback. Market reveal is an audit/calibration layer, never an upstream target.

## Status
- Game Reality: PASS / frozen architecture
- QB Reality: PASS
- Player Reality: PASS structural certification
- 60K Final Blind: PASS
- Football Architecture Freeze: PASS
- Current football input refresh before lock: pending as reports evolve
- Complete FanDuel scoring: next
- FanDuel salary join: pending
- Monster Player Values: pending
- Lineup engine / Monster 150: pending
