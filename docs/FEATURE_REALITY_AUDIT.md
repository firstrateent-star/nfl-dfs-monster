# FEATURE REALITY AUDIT

This ledger is required by `MONSTER_CONSTITUTION.md`. Status is conservative: absence of direct implementation evidence is never promoted by assumption.

Last reset: 2026-09-08 after detecting scope drift between the original Monster design and the Beta22 structural baseline.

| Family | Required Full-Monster role | Current status | Evidence / next gate |
|---|---|---|---|
| Historical team offense | Team identity / scoring / efficiency | ACTIVE | Beta22 uses historical scoring/yards; retain and re-audit provenance. |
| Historical opponent defense | Opponent suppression / game interaction | ACTIVE | Beta22 uses opponent defensive points/yards allowed; retain and re-audit. |
| Historical player performance | Player identity / role / efficiency | PARTIAL | Player pools/roles exist; full mechanism coverage must be inventoried. |
| Finite possessions/play supply | Conserved game opportunity | ACTIVE | Shared finite game/play supply and structural gates passed. |
| Target/rush opportunity conservation | Player allocation | ACTIVE | Existing target/rush world audits and QB reservoir gates passed. |
| Current personnel/starter state | Current football state | ACTIVE/PARTIAL | Explicit current-state overrides exist; freshness/provenance must be refreshed before lock. |
| Health availability/effectiveness/uncertainty | Current player state | ACTIVE/PARTIAL | Three-way health architecture exists; provider freshness seam remains. |
| QB/system continuity | Current team identity | ACTIVE | Continuity-conditioned identity is implemented. |
| Coaching/scheme/coordinator | System reality | PARTIAL | Continuity exists; full coaching/scheme/tendency coverage requires audit. |
| Offensive line | Unit / matchup reality | PARTIAL | Existing state claims OL mechanisms enabled; causal coverage and source audit required. |
| Defensive front / pass rush | Unit / matchup reality | PARTIAL | D/ST/pass disruption exists; full personnel/trait mechanism audit required. |
| Secondary / coverage | Unit / matchup reality | PARTIAL/UNKNOWN | Must trace player/unit coverage evidence and interaction path. |
| Height | Human physical reality | UNKNOWN | Must locate source, coverage and causal use. |
| Weight | Human physical reality | UNKNOWN | Must locate source, coverage and causal use. |
| Wingspan/length | Human physical reality | UNKNOWN/ABSENT | Must locate source or document unavailable fallback. |
| Athletic testing | Human physical reality | UNKNOWN | Speed/agility/explosion testing provenance and causal use required. |
| Age | Biology / career state | UNKNOWN | Must locate and test mechanism. |
| NFL experience | Development/decline/uncertainty | UNKNOWN | Must locate and test mechanism. |
| Madden ratings | Scouting-style player ability evidence | ABSENT/UNVERIFIED | No production implementation was found in direct repository search. This is a BLOCKER for Full Monster promotion until implemented or explicitly replaced by approved evidence-backed scouting traits. |
| QB trait interactions | Play outcome mechanics | PARTIAL | QB rushing calibration exists; passing trait x pressure/coverage mechanics require audit/expansion. |
| WR/TE trait vs coverage | Separation/catch/YAC mechanics | UNKNOWN/PARTIAL | Receiving hierarchy exists; trait-level matchup interaction not certified. |
| RB trait vs front/tackling | Rush efficiency/explosive tail | UNKNOWN/PARTIAL | Rushing anatomy exists; trait-level matchup interaction not certified. |
| Blocking vs front/pass rush | Pressure/rush efficiency | PARTIAL | OL mechanism exists but full player-trait interaction not certified. |
| Weather | Environment | PARTIAL/UNKNOWN | Current-state refresh policy exists; exact causal mechanism/coverage requires audit. |
| Venue/surface | Environment | UNKNOWN | Audit required. |
| Home field | Environment | ACTIVE | Beta22 home-field margin shift exists. |
| Travel/rest | Environment/current circumstance | UNKNOWN | Audit evidence and effect before inclusion. |
| Market blindness | Experimental design | ACTIVE | Existing blind manifests certify market/salary/ownership excluded upstream. |
| Correlated player/game worlds | Joint Sunday reality | ACTIVE | 538-entity 10K artifact persists same-world player/game state. |
| Astrology/birth chart | Experimental micro signal | SHADOW | Must remain zero-authority unless prospective OOS evidence supports promotion. |
| FanDuel scoring/salary/OLR | Downstream DFS | ACTIVE DOWNSTREAM ONLY | Never evidence of football-model completeness. |

## Immediate blocker sequence
1. Inventory actual code/data for every UNKNOWN/PARTIAL row.
2. Build a canonical player-trait schema with provenance and missingness.
3. Acquire/normalize Madden/scouting-style ratings and physical/athletic/age/experience data without using market/DFS information.
4. Implement trait-to-mechanism transforms at the football-play/matchup layer; do not add direct fantasy bonuses.
5. Add mechanism-direction, conservation, missing-data and counterfactual tests.
6. Run ablations: structural baseline vs +physical vs +Madden/scouting vs +current-state/system vs combined candidate.
7. Run multiple blind seeds/world counts.
8. Freeze Full-Reality candidate before Vegas reveal.
9. Compare candidate vs Beta22 baseline vs market vs actual results.
10. Only after promotion rerun FanDuel/OLR and build Monster 150.

## Definition of done
This document has no UNKNOWN rows in required baseline categories; PARTIAL rows are either promoted with evidence or explicitly accepted as documented limitations; Madden/scouting and core physical/athletic families satisfy the Constitution; blind simulation gates pass; and a run manifest records feature-family status/provenance.
