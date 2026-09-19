# MONSTER v7 Build Status

Updated: 2026-09-19

## Frozen control

The immutable scientific control for v7 is:

- architecture: v6.3.8
- commit: `374988076b0a443c1b6eaa368e7641439f078b21`
- branch: `release/v6-verified-baseline`

No v7 experiment may silently mutate this control or reinterpret its Week 1 grade.

## v7 development branch

- branch: `work/v7-reality-engine`
- constitution: `docs/V7_REALITY_ENGINE_CONSTITUTION.md`

## Proven foundation

The following v7 seams have passed their dedicated foundation workflow together with inherited event-conservation, football-state, 11v11 snap and market-blind guards:

### Reality Engine compatibility shell

Files:
- `src/monster/reality/engine.py`
- `src/monster/reality/world_state.py`
- `src/monster/reality/ledger.py`

Properties:
- same-seed v7 compatibility execution produces zero football drift from direct v6 execution
- world state is immutable
- runtime identity is fingerprinted
- market, DFS and direct score inputs remain forbidden upstream
- v6-adapted telemetry is explicitly labeled `V6_ADAPTER` rather than pretending native state fidelity

### Native ledger path

Files:
- `src/monster/reality/native_ledger.py`
- `src/monster/reality/reducers.py`

Native v7 ledger supports:
- exact 11v11 snap records
- pre-snap football state
- post-snap football state
- availability / substitution transition records
- drive terminal records
- game final record

The offensive player box-score reducer reproduces the existing engine's offensive box scores from ledger snap events in parity tests.

### Live availability and substitutions

Files:
- `src/monster/reality/availability.py`
- `src/monster/reality/identity_catalog.py`

Properties:
- live player exits are immutable state transitions
- QB replacement must already exist in the pregame active world
- a replacement QB uses his own compiled `PlayerIdentity`, not cloned starter attributes
- exited receivers are removed from subsequent live identity projection
- transitions can be written to the native event ledger

The architecture is present; no injury / benching probability model has been promoted yet.

### Coherent game-day latent world

Files:
- `src/monster/reality/game_latents.py`
- `src/monster/reality/latent_authority.py`

The sampler produces deterministic per-world, correlated mechanism-level conditions for:
- QB execution
- pass protection
- receiver execution
- run blocking
- ballcarrier execution
- pass rush
- coverage execution
- run fit
- tackling
- special teams execution

These latents currently have **zero production authority**.

The authority registry explicitly limits each factor to named football mechanisms. No latent has score, points, fantasy or DFS jurisdiction.

### QB rush event taxonomy

File:
- `src/monster/reality/qb.py`

Observed QB rushing is separable into:
- sneak
- designed keeper
- pressure scramble
- coverage scramble
- kneel

This taxonomy is audit-only. It does not yet alter generation.

## First v7 football shadow: participation-first opportunity

Files:
- `src/monster/reality/opportunity.py`
- `scripts/runtime_v700_composer.py`
- `scripts/run_reality_loop_v700.py`
- `scripts/audit_full_game_counterfactual_v700.py`
- `scripts/audit_coaching_policy_identity_v700.py`
- `scripts/compare_week1_v638_v700.py`

Hypothesis:

> Once exact snap participants are known, final target / designed-run assignment should not continue using pre-sampled usage share as fallback authority.

Frozen from v6.3.8:
- pregame target participation prior
- pregame rush participation prior
- current roster / health
- exact snap-world construction
- coaching
- matchup identity
- Madden / player execution
- game state
- scoring
- environment
- market blindness

Changed only after participants exist:
- no-evidence target assignment begins from live participants rather than their preallocated usage shares
- historical pass-depth evidence can tilt target assignment among live participants
- historical run-geometry evidence can tilt designed-run assignment among live participants
- QB is not a generic non-sneak designed-run fallback; direct geometry evidence is required
- v6 usage-based opportunity-skill tail is disabled because it would reintroduce post-participation share authority

Status:
- code/lint/inherited causal gates: PASS
- paired Week 1 run: `35448918053`
- control: frozen v6.3.8
- worlds: 120 per game per arm
- common random numbers: yes
- shared pregame inputs: yes
- Week 1 truth loaded after both simulations: yes
- promotion status: SHADOW_NOT_PROMOTED

Promotion requires:
1. hierarchical QB mechanism -> play -> drive -> game counterfactual remains valid,
2. coaching differentiation remains valid,
3. role allocation meaningfully improves or reveals a clearer causal defect,
4. player probabilistic metrics do not materially regress,
5. score anatomy / game ecology do not materially regress,
6. identity propagation does not degrade,
7. all market-blind and conservation gates remain intact.

A Week 1 aggregate improvement alone is insufficient.

## v6.3.9 side evidence

The empirical gadget-entry branch remains a separate v6 closeout experiment:
- branch: `work/v639-gadget-entry-calibration`
- paired closeout run: `35447842743`

v7 may inherit empirical recurrence evidence from v6.3.9 only after that evidence is reviewed. It must not replace the frozen v6.3.8 control by assumption.

## Current architectural boundary

v7 now has the contracts necessary to represent:

```
immutable pregame world
        ↓
coherent game-day latent world
        ↓
live roster / substitution state
        ↓
full player identity catalog
        ↓
exact snap participants
        ↓
participation-first opportunity
        ↓
football interaction / outcome
        ↓
native immutable event ledger
        ↓
drive / box-score reducers
        ↓
DFS later
```

The remaining major work is to migrate production football generation behind these contracts one mechanism at a time.

## Next research sequence

1. Finish the v7 participation-first paired reality test.
2. Promote, revise or reject that mechanism from evidence.
3. Build native QB rush-family generation using historical designed-run / scramble / sneak / kneel evidence.
4. Wire live availability transitions into snap construction using full identity catalogs.
5. Calibrate game-day latent factor distributions and then test one local latent mechanism at a time.
6. Migrate pace / clock / play-volume logic.
7. Repair scoring-channel anatomy through fourth-down, FG, red-zone and non-offensive event mechanisms.
8. Expand rolling historical out-of-sample calibration.
9. Only after football gates pass, re-run the full DFS world matrix / optimizer layer.

## Rule for every v7 upgrade

Do not ask whether a patch makes the Week 1 output look better.

Ask:

> Did a better football mechanism create more realistic distributions for the right causal reason, while preserving all unrelated parts of the machine?
