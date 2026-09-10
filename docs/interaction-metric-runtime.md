# Monster Interaction Metric Runtime

## Center

Monster is a market-blind, stateful NFL simulator. Football conditions, roster state,
players, decisions and interactions create events; events create game outcomes and player
statistics; fantasy value is downstream.

A metric is never allowed to influence the simulator merely because it is available.
It must have a governed evidence definition and an explicit football interaction jurisdiction.

## Two registries

### `config/feature_registry.yaml`

Defines evidence governance:

- feature family
- refresh cadence
- broad jurisdiction
- authority
- production weight
- market-derived/leakage status
- promotion gate where applicable

### `config/interaction_registry.yaml`

Defines football jurisdiction:

- entity scope: player, position group, unit, team, game, environment or coaching
- interaction scale: `1v1`, `small_group`, `unit`, `11v11`, `game_state`
- play phase: `pre_snap`, `initial_contact`, `development`, `resolution`, `post_play`
- owning mechanism(s)
- audit-only status

The interaction registry assigns no coefficient. It only declares where evidence may be
considered.

## Runtime contract

`InteractionMetricRouter` receives metric observations plus one interaction context. It
returns accepted and rejected evidence with explicit rejection reasons.

The router does **not** sum, average or otherwise blend unrelated metrics. The mechanism
that owns an interaction must explicitly interpret routed evidence. This prevents a new
feature from becoming an unrestricted generic projection modifier.

A production-upstream metric is rejected when:

- it is not in the Feature Registry;
- it has no Interaction Registry entry;
- it is market-derived;
- it has zero production authority/weight or is audit-only;
- its entity scope does not match;
- the current interaction scale does not match;
- the play phase does not match;
- the mechanism does not match;
- a participant set is supplied and the metric's entity is not participating.

Missing evidence is therefore naturally neutral.

## Interaction resolution ladder

Use the highest resolution supported by reliable evidence, not the highest resolution that
can be imagined:

1. `game_state` — score, clock, field position, weather, coaching state.
2. `11v11` — personnel package, play call, defensive structure, team-level context.
3. `unit` — offensive line vs front, coverage unit, run front, special teams.
4. `small_group` — combo blocks, stunts, route combinations, coverage exchanges.
5. `1v1` — receiver/defender, blocker/rusher, runner/tackler, catchpoint.

Resolution should increase only when evidence and validation justify it. Lower-resolution
mechanisms remain valid fallbacks when assignment-level data is unavailable.

## Play-time model

Metrics also have a phase so timing can eventually become causal:

- `pre_snap`: availability, substitutions, personnel, formation, play decision.
- `initial_contact`: line engagement, releases, jams, run fits.
- `development`: route separation, rush development, protection, coverage rotation.
- `resolution`: throw/catchpoint, tackle, sack, broken tackle, turnover.
- `post_play`: drive state, fatigue update, substitution response, calibration/audit.

The current engine does not yet simulate continuous physical time. These phases establish a
stable seam for later time-to-pressure, time-to-separation and related mechanisms without
requiring a rewrite.

## Adding a new reality metric

1. **Ingest and provenance** — identify source, timestamp/cadence, entity key and missingness.
2. **Feature Registry** — register family, authority, leakage class and production gate.
3. **Interaction Registry** — state exactly who/what the metric describes, at what scale,
   during what phase and in which mechanism it may act.
4. **Observation compiler** — emit a `MetricObservation` with entity, value, confidence and source.
5. **Mechanism integration** — request the routed metric only inside the owning football
   mechanism and define a bounded interpretation.
6. **Counterfactual test** — show that changing the metric moves only the mechanism it is
   supposed to move.
7. **Conservation test** — ensure event, team and player ledgers still agree.
8. **Historical/OOS gate** — test whether the added resolution improves football reality.
9. **Promotion** — only then grant production authority. Market/DFS information remains
   downstream regardless of apparent predictive usefulness.

## Examples

- Wingspan belongs at a receiver/defender catchpoint, not as a direct fantasy modifier.
- Pass-rush rating belongs in blocker/rusher or front/protection interactions, not in final
  score directly.
- Wind belongs in ball-flight/kicking mechanisms and may be irrelevant to an inside run.
- Active probability belongs before personnel selection; once a player is not on the field,
  his 1v1 traits cannot affect that snap.
- Sportsbook spread belongs in a post-freeze market audit only.

## Current development direction

The next high-value integration is a snap/drive trace that records who was participating,
interaction context and terminal drive outcome. That trace will let us diagnose the current
TD-drive excess and then activate dormant offense/defense evidence in the narrow mechanisms
that actually cause drive progress or failure.
