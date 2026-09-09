from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.feature_compile.units import compile_team_unit_effects
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.league import compile_team_state_map

MATCHUPS = (
    ("CHI", "CAR"), ("BUF", "HOU"), ("NO", "DET"), ("CLE", "JAC"),
    ("TB", "CIN"), ("ATL", "PIT"), ("NYJ", "TEN"), ("BAL", "IND"),
    ("ARI", "LAC"), ("WAS", "PHI"), ("MIA", "LV"), ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _rating(value: float | None, center: float = 78.0, scale: float = 12.0) -> float:
    if value is None:
        return 1.0
    return float(np.clip(1.0 + 0.22 * np.tanh((float(value) - center) / scale), 0.75, 1.25))


def _defensive_unit(players) -> DefensiveUnit:
    front = []; coverage = []
    for p in players:
        if p.defense_snap_share < 0.03: continue
        pos = p.position.upper()
        item = DefensiveIdentity(player_id=p.player_id, name=p.player_id, position=pos, coverage=_rating(p.madden_coverage), pass_rush=_rating(p.madden_pass_rush), run_defense=_rating(p.madden_tackle), tackling=_rating(p.madden_tackle), ball_hawk=_rating(p.madden_coverage))
        if pos in {"DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "MLB"}: front.append(item)
        if pos in {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB"}: coverage.append(item)
    return DefensiveUnit(front=tuple(front), coverage=tuple(coverage))


def _team_identity(team_id: str, pool, reality, unit_players, state) -> TeamIdentity:
    compiled: dict[str, PlayerIdentity] = {}
    for p in pool.players:
        usage = p.qb_pass_share if p.position == "QB" else max(p.target_share, p.rush_share, 0.001)
        inputs = reality.get(p.player_id)
        if inputs is None: compiled[p.player_id] = PlayerIdentity(p.player_id, p.display_name, p.position, usage_weight=usage)
        else: compiled[p.player_id], _ = compile_v13_player_identity(player_id=p.player_id, name=p.display_name, position=p.position, usage_weight=usage, inputs=inputs)
    qbs = [p for p in pool.players if p.position == "QB"]
    if not qbs: raise ValueError(f"{team_id} has no quarterback in current pool")
    qb_state = max(qbs, key=lambda p: p.qb_pass_share); qb = compiled[qb_state.player_id]
    rushers = tuple(PlayerIdentity(compiled[p.player_id].player_id, compiled[p.player_id].name, compiled[p.player_id].position, usage_weight=max(p.rush_share, 0.001), efficiency=compiled[p.player_id].efficiency, explosive=compiled[p.player_id].explosive, turnover_security=compiled[p.player_id].turnover_security) for p in pool.players if p.rush_share > 0.001)
    receivers = tuple(PlayerIdentity(compiled[p.player_id].player_id, compiled[p.player_id].name, compiled[p.player_id].position, usage_weight=max(p.target_share, 0.001), efficiency=compiled[p.player_id].efficiency, explosive=compiled[p.player_id].explosive, turnover_security=compiled[p.player_id].turnover_security) for p in pool.players if p.position in {"RB", "WR", "TE"} and p.target_share > 0.001)
    effects, _ = compile_team_unit_effects(unit_players)
    pass_eff = float(np.clip(1.0 + 0.55 * state.offensive_epa_per_play + state.physical_madden_effect + state.injury_effect + 0.35 * state.weather_effect, 0.72, 1.28))
    rush_eff = float(np.clip(1.0 + 0.35 * state.offense_strength + state.injury_effect + 0.20 * state.weather_effect, 0.75, 1.25))
    return TeamIdentity(team_id=team_id, quarterback=qb, rushers=rushers or (qb,), receivers=receivers, neutral_pass_rate=pool.neutral_pass_rate, pass_efficiency=pass_eff, rush_efficiency=rush_eff, pass_protection=float(np.clip(1.0 + effects.pass_protection_effect, 0.90, 1.10)), run_blocking=float(np.clip(1.0 + effects.run_block_effect, 0.90, 1.10)), field_goal_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)), punt_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--policy", type=Path, required=True); parser.add_argument("--personnel", type=Path, required=True); parser.add_argument("--player-usage", type=Path, required=True); parser.add_argument("--worlds", type=int, default=250); parser.add_argument("--seed", type=int, default=2026090913); parser.add_argument("--out", type=Path, default=Path("artifacts/week1-v13-first-sim")); args = parser.parse_args()
    policy = _read(args.policy); personnel = _read(args.personnel); usage = _read(args.player_usage)
    pools = apply_health_to_skill_pools(compile_current_skill_pools(personnel, usage, policy=policy), personnel); reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE); units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS: states.update(compile_team_state_map(policy, {away: home, home: away}))
    teams = {team: _team_identity(team, pools[team], reality, units[team], states[team]) for pair in MATCHUPS for team in pair}; defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    game_rows=[]; player_acc=defaultdict(lambda: defaultdict(list)); anatomy_acc=defaultdict(lambda: defaultdict(list)); outcome_acc=defaultdict(lambda: {"away_wins":0,"home_wins":0,"ties":0})
    for game_idx,(away,home) in enumerate(MATCHUPS):
        game=f"{away}@{home}"
        for world in range(args.worlds):
            result=simulate_regulation_game(teams[away],teams[home],away_defense=defenses[away],home_defense=defenses[home],seed=args.seed+game_idx*1_000_003+world); assert_event_conservation(result); summary=summarize_game(result)
            for key,value in asdict(summary).items(): anatomy_acc[game][key].append(value)
            if summary.away_points>summary.home_points: outcome_acc[game]["away_wins"]+=1
            elif summary.home_points>summary.away_points: outcome_acc[game]["home_wins"]+=1
            else: outcome_acc[game]["ties"]+=1
            for player_id,box in result.player_stats.items():
                for key,value in asdict(box).items(): player_acc[(game,player_id)][key].append(value)
    for game,metrics in anatomy_acc.items():
        away,home=game.split("@"); row={"game":game,"worlds":args.worlds}
        for key,values in metrics.items():
            arr=np.asarray(values,dtype=float); row[f"{key}_mean"]=float(arr.mean())
            if key in {"away_points","home_points","snaps","scrimmage_plays","drives"}: row[f"{key}_p10"]=float(np.quantile(arr,.10)); row[f"{key}_p50"]=float(np.quantile(arr,.50)); row[f"{key}_p90"]=float(np.quantile(arr,.90))
        row["completion_percentage"]=row["completions_mean"]/row["pass_attempts_mean"] if row["pass_attempts_mean"] else 0.0; row["sack_rate"]=row["sacks_mean"]/row["dropbacks_mean"] if row["dropbacks_mean"] else 0.0; row["scramble_rate"]=row["scrambles_mean"]/row["dropbacks_mean"] if row["dropbacks_mean"] else 0.0; row["interception_rate"]=row["interceptions_mean"]/row["pass_attempts_mean"] if row["pass_attempts_mean"] else 0.0
        outcomes=outcome_acc[game]; row["away"]=away; row["home"]=home; row["total_mean"]=row["away_points_mean"]+row["home_points_mean"]; row["margin_mean"]=row["away_points_mean"]-row["home_points_mean"]; row["away_win_probability"]=outcomes["away_wins"]/args.worlds; row["home_win_probability"]=outcomes["home_wins"]/args.worlds; row["tie_probability"]=outcomes["ties"]/args.worlds; row["projected_winner"]=away if row["away_win_probability"]>row["home_win_probability"] else home; row["projected_away_score"]=round(row["away_points_mean"]); row["projected_home_score"]=round(row["home_points_mean"]); row["projected_score"]=f'{away} {row["projected_away_score"]} - {home} {row["projected_home_score"]}'; row["winner_probability"]=max(row["away_win_probability"],row["home_win_probability"]); game_rows.append(row)
    names={p.player_id:p.display_name for pool in pools.values() for p in pool.players}; positions={p.player_id:p.position for pool in pools.values() for p in pool.players}; player_rows=[]
    for (game,player_id),metrics in player_acc.items():
        row={"game":game,"player_id":player_id,"player":names.get(player_id,player_id),"position":positions.get(player_id,"")}
        for key,values in metrics.items(): arr=np.asarray(values,dtype=float); row[f"{key}_mean"]=float(arr.mean()); row[f"{key}_p90"]=float(np.quantile(arr,.90))
        player_rows.append(row)
    args.out.mkdir(parents=True,exist_ok=True); game_df=pl.DataFrame(game_rows).sort("total_mean",descending=True); game_df.write_csv(args.out/"game_distributions.csv"); pl.DataFrame(player_rows).write_csv(args.out/"player_distributions.csv")
    projection_cols=["game","projected_winner","winner_probability","projected_score","projected_away_score","projected_home_score","away_win_probability","home_win_probability","tie_probability","total_mean","margin_mean","away_points_p10","away_points_p50","away_points_p90","home_points_p10","home_points_p50","home_points_p90"]; game_df.select(projection_cols).write_csv(args.out/"projected_scores_and_outcomes.csv")
    anatomy_cols=["game","scrimmage_plays_mean","pass_plays_mean","pass_attempts_mean","dropbacks_mean","run_plays_mean","rush_attempts_mean","scrambles_mean","sacks_mean","completions_mean","completion_percentage","interceptions_mean","interception_rate","fumbles_lost_mean","sack_rate","scramble_rate","punts_mean","field_goal_attempts_mean","field_goals_made_mean","touchdowns_mean","drives_mean","total_mean"]; game_df.select(anatomy_cols).write_csv(args.out/"football_anatomy.csv")
    manifest={"model":"Monster v1.3 Full-Reality event-by-event shadow","week":1,"season":2026,"worlds_per_game":args.worlds,"seed":args.seed,"games":len(MATCHUPS),"market_blind_football":True,"scoreboard_event_derived":True,"player_stats_event_derived":True,"definition_safe_anatomy":True,"defensive_identity_active":True,"full_reality_player_bridge_active":True,"unit_bridge_active":True,"projection_summary":"projected_scores_and_outcomes.csv","football_anatomy":"football_anatomy.csv","promotion_status":"SHADOW_FIRST_SIMULATION_NOT_PROMOTED"}; (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(game_df.select(["game","projected_winner","winner_probability","projected_score"])); print(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
