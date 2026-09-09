# FEATURE REALITY AUDIT

This ledger is required by `MONSTER_CONSTITUTION.md`. Status is conservative: source presence alone does not prove causal authority, and passing structural tests does not erase documented limitations.

Last evidence reconciliation: 2026-09-09 after the independent Full-Reality 60K certification runs.

| Family | Required Full-Monster role | Current status | Evidence / remaining gate |
|---|---|---|---|
| Historical team offense | Team identity / scoring / efficiency | ACTIVE | 2025 market-blind team policy priors compiled for all 32 teams. |
| Historical opponent defense | Opponent suppression / game interaction | ACTIVE | Opponent interactions remain upstream of scoring and market-blind. |
| Historical player performance | Player identity / role / efficiency | ACTIVE/PARTIAL | Current player usage, snap priors and defensive history are compiled; some no-history players rely on current depth/trait evidence. |
| Finite possessions/play supply | Conserved game opportunity | ACTIVE | Shared finite game/play supply; broad Player Reality gate passes. |
| Target/rush opportunity conservation | Player allocation | ACTIVE | World-level target/rush rank and opportunity audits pass at 60K/game. |
| Current personnel/starter state | Current football state | ACTIVE/PARTIAL | 2026 canonical roster/depth state is active; freshness must still be refreshed before slate lock. |
| Health availability/effectiveness/uncertainty | Current player state | ACTIVE/PARTIAL | Three-way health mechanism is active and tested. Current provider returned 0 formal Week-1 injury rows, so 13 explicit overrides carry current non-roster health evidence until official reports populate. |
| QB/system continuity | Current team identity | ACTIVE | Continuity-conditioned policy active for 32 teams; 2022-2025 OOS gate previously passed. |
| Coaching/scheme/coordinator | System reality | PARTIAL | Historical policy + continuity encode system persistence, but a fully current named coach/coordinator/scheme evidence layer is not yet certified. |
| Offensive line | Unit / matchup reality | ACTIVE/PARTIAL | 419 relevant OL; 360 with Madden pass/run blocking; historical OL outcomes feed unit compilation. Two teams lack historical OL outcome prior and use bounded fallback. |
| Defensive front / pass rush | Unit / matchup reality | ACTIVE/PARTIAL | 1,122 relevant defenders; 568 front players with Madden pass-rush evidence; compiled unit interaction is active. Coverage is not universal. |
| Secondary / coverage | Unit / matchup reality | ACTIVE/PARTIAL | 692 coverage players have Madden coverage evidence; coverage unit gate passes, with explicit missingness for uncovered players. |
| Height | Human physical reality | ACTIVE | Promotion population coverage 518/518; causal player mechanism path certified. |
| Weight | Human physical reality | ACTIVE | Promotion population coverage 518/518; physical mechanism path enabled. |
| Wingspan/length | Human physical reality | PARTIAL/ABSENT | Mechanism schema supports wingspan, but no sufficiently covered canonical Week-1 source is certified. Missingness remains neutral. This is a documented Full-Reality limitation, not fabricated data. |
| Athletic testing | Human physical reality | ACTIVE/PARTIAL | Forty evidence exists for 306/518 promotion skill players (59.1%); missing testing remains neutral and Madden speed/acceleration provides separate scouting evidence. |
| Age | Biology / career state | ACTIVE | Birth date coverage 518/518 promotion skill population. Age alone has zero production penalty by test. |
| NFL experience | Development/decline/uncertainty | ACTIVE/PARTIAL | Years of experience feeds a conservative workload proxy only for uncertainty; it is not represented as exact career workload. |
| Madden ratings | Scouting-style player ability evidence | ACTIVE | Madden 27 ratings loaded as bounded scouting proxy with no direct fantasy authority. Promotion skill coverage: speed/accel 483/518 (93.2%), route/catch 480/518 (92.7%). Defensive, OL and specialist source gates pass. |
| QB trait interactions | Play outcome mechanics | ACTIVE/PARTIAL | QB state, rushing reservoir and broader team mechanisms are active; richer throw-trait x pressure/coverage play-resolution remains an expansion target. |
| WR/TE trait vs coverage | Separation/catch/YAC mechanics | ACTIVE/PARTIAL | Route/catch/speed/catchpoint mechanisms are active and counterfactually tested; matchup resolution remains aggregate/unit-level rather than a complete defender-by-receiver route engine. |
| RB trait vs front/tackling | Rush efficiency/explosive tail | ACTIVE/PARTIAL | Physical/Madden player mechanisms and front/tackling unit evidence are active; individual blocker/defender assignment is not fully play-resolved. |
| Blocking vs front/pass rush | Pressure/rush efficiency | ACTIVE/PARTIAL | Madden OL + historical OL outcome priors compile into unit effects; individual trench assignments remain aggregate. |
| Weather | Environment | ACTIVE/PARTIAL | Week-1 environment snapshot is loaded; dome nullification and adverse-weather direction tests pass. Forecast must refresh before lock. |
| Venue/surface | Environment | PARTIAL | Venue/roof/surface are recorded. Roof/dome affects weather authority; surface itself does not yet have a separately validated causal production mechanism. |
| Home field | Environment | ACTIVE | Home-field state remains active. |
| Travel/rest | Environment/current circumstance | ABSENT/PARTIAL | No separately certified travel/rest mechanism. Must remain neutral until sourced and prospectively tested. |
| Market blindness | Experimental design | ACTIVE | 60K manifests certify market/contaminated references false and salary/ownership excluded upstream. |
| Correlated player/game worlds | Joint Sunday reality | ACTIVE | Same-world finite game/player architecture and broad allocation gates pass. |
| Astrology/birth chart | Experimental micro signal | SHADOW | Trace-only; explicit counterfactual test proves zero production authority. |
| FanDuel scoring/salary/OLR | Downstream DFS | ACTIVE DOWNSTREAM ONLY | Never evidence of football-model completeness; Monster150 remains paused until promotion. |

## Current certification evidence

- Full-Reality production suite: 112/112 tests pass.
- Dedicated causal counterfactual suite: 7/7 passes.
- Full-Reality feature/source coverage gate: PASS.
- Broad Player Reality structural gate: 20/20 PASS at 60,000 worlds/game.
- QB rushing calibration gate: PASS at 60,000 worlds/game.
- Market/DFS blindness gate: PASS.
- Independent 60K seeds `2026090902` and `2026090904` are highly stable: across 12 games, mean absolute seed-to-seed change is about 0.100 points for game totals and 0.073 points for margins; maximum absolute changes are 0.248 and 0.141 points respectively.
- Seed `2026090904` artifact: GitHub Actions run `34314517218`, artifact `10089726381`, SHA-256 `2ad5889ad798902b1a230bae55ca8732fd87cdee55614a6af36e12fc4328eb84`.

## Remaining promotion gates

1. Build and persist a formal ablation ladder: structural baseline → +human/physical → +Madden/scouting → +unit interactions → +current state/system → +environment → combined Full Reality.
2. Verify that each promoted family changes only intended football mechanisms and does not violate conservation or create implausible tails.
3. Explicitly accept or resolve documented limitations: wingspan/length coverage, travel/rest, current named coaching/scheme depth, surface-specific mechanism, health-provider freshness, and aggregate rather than assignment-level matchup resolution.
4. Refresh current health/personnel/weather immediately before slate lock.
5. Freeze the Full-Reality candidate before any new market reveal/comparison.
6. Only after promotion rerun FanDuel correlated worlds/OLR and Monster150.

## Definition of done

Required families are either ACTIVE with provenance + causal evidence or explicitly accepted as bounded documented limitations; the formal ablation ladder passes; independent-seed stability passes; market blindness is frozen; and `MONSTER_STATE.md` records a reproducible Football Reality v1 release manifest. No feature may be promoted merely because a source column exists.
