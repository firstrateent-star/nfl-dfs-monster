# MONSTER v7 — Reality Engine Constitution

## Thesis

v6 proved that MONSTER can generate football causally. v7 must stop treating role shares, isolated play outcomes and versioned runtime patches as the center of the system.

The v7 center is a **hierarchical football world**:

```
long-lived football identity
        ↓
pregame uncertain world
        ↓
game-day latent performance world
        ↓
drive state
        ↓
snap personnel + assignments
        ↓
player-v-player interactions
        ↓
football event
        ↓
state transition
        ↓
conserved ledger
        ↓
box score
        ↓
DFS observation
```

Opportunity should increasingly be an *emergent consequence of participation, assignments, play design and game state*, not a preallocated answer.

## Non-negotiable constitution

1. **Football first.** Salary, ownership, Vegas and fantasy projections cannot influence football generation.
2. **No result fitting.** Realized game scores or fantasy outcomes are grading truth, never direct football inputs.
3. **Event-derived reality.** Scores, stats and DFS points must reconcile to the same event ledger.
4. **One runtime.** Production, audits, counterfactuals and benchmarks execute the same engine.
5. **Explicit uncertainty.** Availability, role, game-day performance and in-game transitions are modeled states, not hidden corrections.
6. **Local mechanisms.** Player traits influence football through specific interactions.
7. **Causal promotion.** A feature is promoted only after mechanism -> play -> drive -> game tests plus out-of-sample reality grading.
8. **Frozen controls.** Every major v7 experiment is paired against an immutable v6 or v7 control with common random numbers.

## Architectural break from v6

v6 accumulated research layers such as `reality_v62`, `current_role_guard_v635/v636/v638/v639` and version-specific runtime composers. Those layers were valuable because they let us isolate mechanisms.

v7 should consolidate them into one explicit engine instead of continuing the patch stack.

Proposed package:

```
monster/reality/
  engine.py
  world_state.py
  pregame.py
  availability.py
  roles.py
  coaching.py
  personnel.py
  assignments.py
  snap.py
  matchups.py
  qb.py
  rushing.py
  passing.py
  resolution.py
  transition.py
  clock.py
  special_teams.py
  ledger.py
  telemetry.py

monster/calibration/
  historical_priors.py
  role_recurrence.py
  game_latents.py
  mechanism_fits.py
  covariance.py

monster/benchmark/
  causal.py
  rolling_oos.py
  week1_2026.py
  calibration.py
  dependence.py
```

DFS remains downstream in `monster/dfs/`.

## World hierarchy

### 1. Identity state

Slow-moving evidence:
- Madden / physical / technical traits,
- historical efficiency,
- coaching identity,
- scheme,
- unit quality,
- player matchup capability,
- continuity and experience.

This is capability, not a predicted box score.

### 2. Pregame world

Sample once per simulation world:
- game-day active roster,
- starter probabilities,
- injury effectiveness,
- depth / rotation,
- OL combinations,
- defensive packages,
- specialist availability,
- weather and field conditions.

The pregame world establishes who can plausibly participate and how the team expects to use them.

### 3. Game-day latent performance world

Sample coherent, historically calibrated residual states such as:
- QB execution day,
- pass-protection cohesion,
- pass-rush effectiveness,
- receiver / coverage execution,
- run-block / run-fit execution,
- tackling,
- offensive pace / cohesion,
- special-teams execution.

These latents must enter local football mechanisms. They may not directly add points or fantasy production.

Purpose: preserve correlated good/bad football days so shootouts, collapses, ceiling games and team-wide failures can emerge without independent per-play noise washing them out.

### 4. Drive state

Carries:
- score / time,
- field position,
- possession,
- timeouts,
- fatigue,
- recent play sequence,
- personnel stress,
- tactical adaptation,
- two-minute / four-minute modes.

### 5. Snap state

For every snap:
- offensive personnel package,
- defensive personnel package,
- exact participants,
- formations / alignment abstractions,
- protection / rush structure,
- coverage / box structure,
- play family,
- assignments,
- matchup graph,
- player fatigue.

This state should be observable in telemetry.

## Major v7 mechanism upgrades

### Participation -> opportunity

Current target and rush history become priors on:
- package participation,
- route participation,
- designed touches,
- assignment likelihood,
- situational usage.

Targets and carries emerge after the snap exists.

This should naturally produce:
- workhorse games,
- committee games,
- receiver disappearances,
- target funnels,
- role replacement after injury,
- low-volume surprise contributors,
without forcing final shares.

### QB rush decomposition

Model separate event families:
- designed QB run,
- read-option keep,
- sneak,
- scramble caused by pressure,
- scramble caused by coverage / open grass,
- kneel.

Sacks remain pass events and are never silently reclassified as rushing.

Each family gets its own historical prior, player mechanism and game-state authority.

### In-game availability and substitution

Add transition hazards for:
- injury / aggravation,
- performance benching where historically appropriate,
- concussion / medical exit,
- rotation changes,
- garbage-time replacement,
- QB handoff.

A player exit updates eligible personnel and downstream opportunity on subsequent snaps. This is required to represent worlds like the Week-1 Murray/Wentz split.

### Pace and play volume

Explicitly model:
- huddle / no-huddle,
- between-play clock,
- two-minute offense,
- hurry-up after negative game states,
- four-minute offense,
- spikes,
- kneels,
- out-of-bounds clock effects,
- timeout use.

Play volume should emerge from both teams' behavior, not be held near a narrow central value.

### Scoring-channel anatomy

v7 must preserve v6's good total ecology while fixing how points are created.

Upgrade:
- fourth-down decision policy,
- field-goal attempt decision,
- kicker range / accuracy and weather,
- red-zone / goal-to-go play ecology,
- turnover return geometry,
- punt / kickoff return TD channels,
- blocked-kick channels,
- safeties.

The target is not a prescribed total. The target is correct causal anatomy across historical holdouts.

### Defensive tactical state

Consolidate existing v6 interaction work into explicit tactical choices:
- shell,
- man / zone abstraction,
- blitz / simulated pressure,
- rush package,
- box count,
- run fit,
- coverage help / bracket,
- protection response.

The defense should change the graph of player interactions on the snap, not merely supply a team multiplier.

## Native event ledger

Every snap should emit an immutable structured record containing:

- world / game / drive / snap IDs,
- pre-snap football state,
- offensive / defensive participants,
- personnel and assignments,
- coaching choice,
- matchup observations,
- pressure path,
- QB decision,
- target / rusher,
- air yards / YAC / run geometry,
- tackles / turnovers,
- special-teams event,
- score change,
- clock transition,
- post-snap state,
- causal telemetry fields.

All box scores, drive summaries, game scores and DFS results are reductions of this ledger.

## Validation stack

v7 is not accepted because Week 1 looks better.

Validation occurs at multiple layers:

| Layer | Examples |
|---|---|
| Structural | conservation, 11v11 uniqueness, legal transitions, participant eligibility |
| Mechanism | pressure, QB response, target selection, run entry, FG decisions |
| Play | completion, sack, scramble, explosive, run geometry |
| Drive | survival, first downs, red-zone entry, terminal type |
| Game | scoring anatomy, pace, turnovers, punts, margins, tails |
| Player | opportunity, box-score distributions, role concentration |
| Dependence | QB-WR, QB-opponent, RB-game-script, DST-opponent relationships |
| Calibration | CRPS, log score, PIT / rank histograms, interval coverage |
| Causal | common-seed intervention response |
| OOS | rolling historical as-of replays plus untouched current weeks |

Week 1 2026 remains a fixed case study, not the optimization target.

## Initial v7 research priorities

P0 — preserve the v6 constitution and build the clean engine shell with ledger compatibility.

P1 — world hierarchy and coherent game-day latent states.

P2 — participation-first personnel / assignment generation and emergent targets / carries.

P3 — QB rush family decomposition and in-game substitution / injury transitions.

P4 — pace / clock / play-volume expansion.

P5 — field-goal, fourth-down, red-zone and non-offensive scoring anatomy.

P6 — rolling historical OOS harness and dependence calibration.

P7 — only after football gates pass, re-run the complete DFS world matrix and lineup layer.

## Success definition

v7 succeeds when it does not merely produce more accurate averages.

It succeeds when changing a real football cause changes the right downstream football consequences, naturally occurring teams and players retain their distinct identities through complete games, extreme outcomes arise from coherent worlds, probability intervals calibrate out-of-sample, and the DFS layer discovers those worlds without influencing them.

The goal is not to predict the Week-1 box score.

The goal is to build a machine from which a Week-1-like box score could plausibly emerge for the right football reasons.
