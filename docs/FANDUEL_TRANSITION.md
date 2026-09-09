# FanDuel Transition Contract

Football architecture is frozen at Monster Football v1 Week 1, blind run `34292058208`.

## Boundary
FanDuel salary, IDs, roster rules, ownership, and optimizer outputs are downstream-only. None may alter football mechanisms or the frozen blind football artifact.

## Scoring order
1. Preserve football world outputs.
2. Attribute player-level interceptions thrown and fumbles lost inside worlds.
3. Apply complete FanDuel scoring to each player/world.
4. Validate scoring conservation and distribution summaries.
5. Join FanDuel slate identity + salary.
6. Compute salary-relative value and threshold probabilities.
7. Add ownership only after independent football/value distributions exist.
8. Solve legal lineups inside simulated Sundays and measure optimal-lineup rate.
9. Build Monster 150 as a portfolio over possible Sundays rather than 150 near-clone projections.

## Current blocker
The frozen 60K artifact already contains FanDuel scoring before turnover penalties. Player-level interception/fumble-lost attribution is not yet modeled, so those partial scores must not be labeled final FanDuel projections.

## FanDuel slate evidence
A prior Week 1 Monster output in the user's File Library preserves FanDuel contest IDs and salaries for contest prefix `133104`, but it is a derived v0.2 output rather than the canonical raw FanDuel upload. It may be used as identity/salary recovery evidence if necessary, but the raw FanDuel player-list CSV remains preferred for the production salary join.

## Next engineering seam
Extend world-level player stats with turnover attribution without changing team/game football distributions. This is a downstream attribution/scoring seam, not permission to retune Game Reality.
