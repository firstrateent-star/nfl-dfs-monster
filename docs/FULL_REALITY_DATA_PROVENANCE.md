# Full Reality data provenance

This file is part of the anti-drift contract in `MONSTER_CONSTITUTION.md`.

## Authority hierarchy
1. Observed NFL play/game/player evidence and current official personnel state.
2. Current roster/depth/injury/participation evidence.
3. Stable physical/biographical facts and measured athletic testing.
4. Madden 27 ratings as bounded scouting-style proxy evidence.
5. Explicit uncertainty/fallbacks where data is missing.
6. Shadow-only experimental signals.

Sportsbook lines, DFS salary, ownership and optimizer output have zero authority upstream of football freeze.

## nflverse / historical football
Used for historical team/player performance, rosters/player identity, snap counts, depth/injury feeds, combine/physical measurements and defensive/player history where available. Historical seasons and current snapshot dates are versioned by workflow manifests.

## Madden 27
Adapter: `src/monster/ingest/madden_players.py`.
Public source mirror: `zachxwalton/madden-ratings-breakdown`, file `scraper/output/madden27_ratings.csv`. The mirror states it scrapes EA's ratings site. The repository does not treat Madden as ground truth.

Current source schema includes player identity/position/team plus speed, acceleration, agility, awareness, ball-carrier vision, block shed, break tackle, carry, catch/CIT/spectacular catch, finesse/power moves, hit power, kick accuracy/power/return, man/zone coverage, pass/run blocking, play recognition, pursuit, release, route running, strength, tackle, QB accuracy/power/pressure and related fields.

Madden jurisdiction is bounded:
- QB traits -> team skill/quarterback passing-quality mechanism.
- RB/WR/TE traits -> team skill talent and player explosive/catch/rushing mechanisms.
- OL traits -> pass protection/run blocking.
- front-seven traits -> pass rush/run-front/tackling proxies.
- secondary traits -> coverage/tackling proxies.
- specialists -> special-teams proxy where current simulation mechanism has jurisdiction.

No Madden rating directly awards fantasy points.

## Physical / athletic / biology
Stable player height, weight, birth date and NFL experience come from player/roster identity data. Combine evidence supplies forty/athletic measurements where available. Missing combine values remain missing; Madden speed can supplement the speed mechanism but does not overwrite observed testing.

Wingspan/length is accepted by the mechanism compiler when available but current canonical source coverage is incomplete. Missing wingspan remains neutral rather than guessed.

Age does not directly reduce expected production. Age plus experience/workload can widen biology/role uncertainty; current injury/effectiveness evidence has higher authority.

## Current health and role
Health is separated into availability, effectiveness-if-active and uncertainty. Current provider injury rows plus explicit dated overrides are versioned. Depth, snap priors and current participation inference determine game-day role evidence.

## Units and matchup interaction
`feature_compile/units.py` and the game kernel preserve position jurisdiction. OL/pass rush/coverage/run-front/skill/special-teams evidence is snap-weighted and bounded. Game simulation consumes offense-vs-opponent unit states, so pass protection interacts with opponent rush and coverage rather than producing independent fantasy bonuses.

## Environment
Versioned Week 1 snapshot: `config/environment/week1_2026_2026-09-09.csv`. Dome/covered venues are weather-neutral. Outdoor temperature/wind/precipitation are applied only through bounded football environment mechanisms. Forecasts must be refreshed before lock because they are time-varying evidence.

## Coaching / continuity
Historical policy plus continuity-conditioned policy is retained. Coaching/system entropy and continuity are football-state inputs; they are not inferred from market movement.

## Experimental shadow layer
Astrology/birth-chart signals have zero production authority. They may be recorded only in isolated traces/experiments until prospective out-of-sample evidence supports a governance decision.
