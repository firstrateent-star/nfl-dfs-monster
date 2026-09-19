# MONSTER v6 — Verified Baseline Closeout

## Status

This branch freezes the last fully Week-1 reality-verified v6 runtime:

- version: v6.3.8
- commit: `374988076b0a443c1b6eaa368e7641439f078b21`
- purpose: immutable scientific control for v7
- football market-blind: yes
- fantasy inputs into football: no
- direct score adjustment: no
- Week 1 truth used before simulation: no

v6.3.9 remains a final shadow experiment testing a more empirical WR/TE designed-rush entry prior. It must not silently replace this control unless its final common-seed closeout benchmark earns promotion.

## What v6 proved

v6 crossed the architectural threshold from a projection-shaped simulator into an auditable causal football engine.

The production path now supports:

1. pregame roster / health / current-role evidence,
2. team coaching and situational play-calling identity,
3. explicit snap participants and player-v-player interactions,
4. Madden / player traits reaching live football mechanisms,
5. event-derived drives, scoring and player box scores,
6. conserved football statistics,
7. DFS scoring strictly downstream,
8. one authoritative production runtime composer / fingerprint,
9. common-random-number causal counterfactuals,
10. truth-after-simulation reality benchmarking.

The strongest causal evidence is the v6.3.8 QB intervention test. A positive QB execution intervention improved read quality for 24/24 teams, yards per drive for 24/24, red-zone entry for 24/24, points for 23/24 and passing yards for 23/24 without any direct score or fantasy adjustment. That establishes a real mechanism -> play -> drive -> game causal chain.

## Week 1 2026 evidence

### Game ecology

| Metric | v6.3.8 | Reference |
|---|---:|---:|
| Sim points / game | 46.53 | 2025 NFL: 46.03 |
| Punts / game | 7.24 | 2025 NFL: 7.11 |
| Turnovers / game | 2.22 | 2025 NFL: 2.31 |
| Explosive 15+ plays / game | 12.89 | 2025 NFL: 12.59 |
| Explosive 20+ plays / game | 6.98 | 2025 NFL: 6.96 |
| Week 1 team-points MAE | 8.86 | — |
| Week 1 game-total MAE | 14.15 | — |
| Week 1 margin MAE | 11.97 | — |
| Week 1 winner accuracy | 58.3% | descriptive only |

These results show that league-level football ecology is increasingly realistic, but a correct aggregate total does not imply correct game anatomy.

### Scoring anatomy

| Channel | v6.3.8 | 2025 NFL |
|---|---:|---:|
| Offensive TD / game | 5.37 | 4.86 |
| Non-offensive TD / game | 0.11 | 0.27 |
| FG attempts / game | 3.35 | 4.00 |
| FG made / game | 2.79 | 3.42 |

v6 reaches approximately the correct scoring mean through the wrong mixture: too much offensive-TD scoring and too little field-goal / non-offensive scoring. v7 must fix the mechanisms that create scoring channels rather than tune total points.

### Player / distribution evidence

- 266 active Week-1 players matched; only one unmatched active truth player was a punter.
- Zero actual rushing opportunities and zero actual targets belonged to an unmodeled skill player.
- FanDuel MAE: 4.52.
- FanDuel CRPS: 3.30.
- P10-P90 player interval coverage: 71.4% versus 80% nominal.
- P05-P95 coverage: 82.0% versus 90% nominal.
- 14.3% of realized player outcomes exceeded simulated P95 versus 5% nominal.

The universe coverage is a v6 strength. Distribution calibration is not. The Week-1 evidence indicates missing coherent ceiling and downside worlds rather than a need to widen fantasy projections directly.

### Role evidence

- rushing team-share TVD: 0.249
- target team-share TVD: 0.286
- rushing active-player-share MAE: 0.0945
- target active-player-share MAE: 0.0583

v6.3.8 improved rushing allocation against the frozen v6.3.6 control while leaving target allocation unchanged, which is evidence that the isolated role experiment behaved causally.

Remaining role misses show that workloads can still be too diffuse. Examples from Week 1 include Jonathan Taylor, David Montgomery, Jordan Mason and Omarion Hampton receiving substantially less simulated rush concentration than realized, while several reserve players retained non-trivial plan mass. This should be addressed structurally through participation and assignment generation, not by fitting Week-1 shares.

### QB rushing

For Week-1 primary QBs that were active in both truth and simulation, v6.3.8 was approximately:

- +3.6 QB carries / game,
- +21.1 QB rushing yards / game.

v7 should explicitly separate designed QB runs, pressure scrambles, kneels and other QB rush families. A single upstream QB-rush reservoir is no longer adequate.

### Identity propagation

v6 has genuine team, coaching, matchup and player identity. It also still attenuates natural identity before final expected game distributions.

Examples from the v6.3.8 propagation audit:

- pass-efficiency SD: 0.108
- rush-efficiency SD: 0.082
- team expected-points SD: 2.78
- QB-efficiency -> QB passing-yards correlation: 0.259
- simple pass/rush efficiency -> team-points correlations near zero

This does not justify tuning to one realized Week-1 slate. The causal intervention tests prove that mechanisms work; the next problem is preserving naturally occurring identity through the complete game.

### Missing world transitions

Week 1 exposed states that v6 cannot represent cleanly. The clearest example is a mid-game QB handoff: the real MIN game split work between Kyler Murray and Carson Wentz, while the pregame v6 world effectively committed to one QB role. v7 needs in-game availability, injury / exit hazards, substitutions and downstream role redistribution.

## v6 closeout doctrine

Do not add more broad v6 coefficient patches.

A v6 change may be retained only if it is:
- isolated,
- leakage-resistant,
- causally interpretable,
- tested with common random numbers,
- graded after simulation against untouched truth,
- non-regressive on conserved football and market-blindness.

Everything larger belongs in v7.

## What v7 inherits unchanged

v7 must preserve these constitutional invariants:

- football reality before DFS,
- market / salary / ownership firewall,
- pregame-as-of provenance,
- score and stats derived from football events,
- 11-v-11 participant truth,
- Madden / player traits only through football mechanisms,
- current health / roster truth,
- coaching and matchup identity,
- conserved event ledger,
- common-seed causal testing,
- runtime fingerprinting,
- truth used for grading only after simulation.

The v6 baseline exists so every v7 architecture change can be measured against a known machine rather than against memory.
