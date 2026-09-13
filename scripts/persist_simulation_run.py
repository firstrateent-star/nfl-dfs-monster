from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote

import polars as pl
import requests

DEFAULT_SUPABASE_URL = "https://nuyvuwqvsgopapjbnndu.supabase.co"
DEFAULT_BUCKET = "monster-simulation-runs"
SCHEMA = "monster"
BATCH_SIZE = 200


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _jsonable(row: dict) -> dict:
    return {str(key): _clean(value) for key, value in row.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pl.read_csv(path).height
        if suffix == ".parquet":
            return pl.read_parquet(path).height
    except Exception:
        return None
    return None


class SupabaseWriter:
    def __init__(self, url: str, key: str) -> None:
        self.url = url.rstrip("/")
        self.key = key
        self.session = requests.Session()

    def _headers(self, *, content_type: str = "application/json") -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Content-Type": content_type,
        }
        # Legacy service_role keys are JWTs and Storage/PostgREST accept them as Bearer tokens.
        # New sb_secret keys must be sent as an API key rather than parsed as a JWT.
        if not self.key.startswith("sb_secret_"):
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def upsert(self, table: str, rows: list[dict], conflict: tuple[str, ...]) -> None:
        if not rows:
            return
        endpoint = f"{self.url}/rest/v1/{table}?on_conflict={','.join(conflict)}"
        headers = self._headers()
        headers.update(
            {
                "Content-Profile": SCHEMA,
                "Prefer": "resolution=merge-duplicates,return=minimal",
            }
        )
        for start in range(0, len(rows), BATCH_SIZE):
            batch = [_jsonable(row) for row in rows[start : start + BATCH_SIZE]]
            response = self.session.post(endpoint, headers=headers, json=batch, timeout=90)
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Supabase upsert failed for {table}: {response.status_code} {response.text[:800]}"
                )

    def upload(self, bucket: str, storage_path: str, source: Path) -> None:
        encoded = "/".join(quote(part, safe="") for part in storage_path.split("/"))
        endpoint = f"{self.url}/storage/v1/object/{quote(bucket, safe='')}/{encoded}"
        content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        headers = self._headers(content_type=content_type)
        headers["x-upsert"] = "true"
        with source.open("rb") as handle:
            response = self.session.post(endpoint, headers=headers, data=handle, timeout=180)
        if response.status_code >= 300:
            raise RuntimeError(
                f"Storage upload failed for {storage_path}: {response.status_code} {response.text[:800]}"
            )


def _team_map(box_dir: Path) -> dict[str, dict[str, str]]:
    path = box_dir / "offensive_player_box_score_distributions.csv"
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    return {
        str(row["player_id"]): {
            "team": str(row.get("team") or ""),
            "game": str(row.get("game") or ""),
            "player": str(row.get("player") or row["player_id"]),
            "position": str(row.get("position") or ""),
        }
        for row in frame.to_dicts()
    }


def _skill_map(sim_dir: Path) -> dict[str, dict]:
    path = sim_dir / "player_skill_snapshot.csv"
    if not path.exists():
        return {}
    return {str(row["player_id"]): row for row in pl.read_csv(path).to_dicts()}


def _defender_map(sim_dir: Path) -> dict[str, dict]:
    path = sim_dir / "defender_skill_snapshot.csv"
    if not path.exists():
        return {}
    return {str(row["player_id"]): row for row in pl.read_csv(path).to_dicts()}


def _opponent_strength_by_player_family(sim_dir: Path) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    defenders = _defender_map(sim_dir)
    if not defenders:
        return {}
    rows: list[dict] = []
    sources = (
        ("same_world_pass_throws.csv", "target_id", "receiving", "complete"),
        ("same_world_designed_runs.csv", "rusher_id", "designed_run", None),
        ("same_world_scrambles.csv", "player_id", "scramble", None),
    )
    for filename, creator_col, family, required_bool in sources:
        path = sim_dir / filename
        if not path.exists():
            continue
        frame = pl.read_csv(path)
        if required_bool is not None and required_bool in frame.columns:
            frame = frame.filter(pl.col(required_bool))
        for row in frame.select([creator_col, "primary_defender_id"]).to_dicts():
            creator = row.get(creator_col)
            defender_id = row.get("primary_defender_id")
            if creator is None or defender_id is None:
                continue
            defender = defenders.get(str(defender_id))
            if defender is None:
                continue
            rows.append(
                {
                    "player_id": str(creator),
                    "play_family": family,
                    "coverage": float(defender.get("coverage", 1.0)),
                    "tackling": float(defender.get("tackling", 1.0)),
                }
            )
    if not rows:
        return {}
    grouped = (
        pl.DataFrame(rows)
        .group_by(["player_id", "play_family"])
        .agg(
            pl.col("coverage").mean().alias("coverage"),
            pl.col("tackling").mean().alias("tackling"),
        )
    )
    return {
        (str(row["player_id"]), str(row["play_family"])): (
            float(row["coverage"]),
            float(row["tackling"]),
        )
        for row in grouped.to_dicts()
    }


def _game_rows(sim_dir: Path, run_id: str) -> list[dict]:
    frame = pl.read_csv(sim_dir / "game_distributions.csv")
    rows = []
    for source in frame.to_dicts():
        rows.append(
            {
                "run_id": run_id,
                "game_id": str(source["game"]),
                "away_team": source.get("away"),
                "home_team": source.get("home"),
                "worlds": int(source.get("worlds") or 0),
                "away_points_mean": source.get("away_points_mean"),
                "home_points_mean": source.get("home_points_mean"),
                "total_mean": source.get("total_mean"),
                "margin_mean": source.get("margin_mean"),
                "away_win_probability": source.get("away_win_probability"),
                "home_win_probability": source.get("home_win_probability"),
                "overtime_probability": source.get("overtime_probability"),
                "scrimmage_plays_mean": source.get("scrimmage_plays_mean"),
                "drives_mean": source.get("drives_mean"),
                "punts_mean": source.get("punts_mean"),
                "field_goal_attempts_mean": source.get("field_goal_attempts_mean"),
                "field_goals_made_mean": source.get("field_goals_made_mean"),
                "touchdowns_mean": source.get("touchdowns_mean"),
                "offensive_touchdowns_mean": source.get("offensive_touchdowns_mean"),
                "defensive_touchdowns_mean": source.get("defensive_touchdowns_mean"),
                "special_teams_touchdowns_mean": source.get("special_teams_touchdowns_mean"),
                "turnovers_mean": (source.get("interceptions_mean") or 0.0)
                + (source.get("fumbles_lost_mean") or 0.0),
                "turnovers_on_downs_mean": source.get("turnovers_on_downs_mean"),
                "explosive_15_mean": source.get("explosive_15_mean"),
                "explosive_20_mean": source.get("explosive_20_mean"),
                "explosive_40_mean": source.get("explosive_40_mean"),
                "explosive_60_mean": source.get("explosive_60_mean"),
                "completion_percentage": source.get("completion_percentage"),
                "sack_rate": source.get("sack_rate"),
                "scramble_rate": source.get("scramble_rate"),
                "payload": {
                    key: _clean(value)
                    for key, value in source.items()
                    if key.endswith("_sd") or key.endswith("_p10") or key.endswith("_p50") or key.endswith("_p90")
                },
            }
        )
    return rows


def _player_rows(sim_dir: Path, box_dir: Path, run_id: str) -> list[dict]:
    frame = pl.read_csv(sim_dir / "player_distributions.csv")
    teams = _team_map(box_dir)
    rows = []
    for source in frame.to_dicts():
        player_id = str(source["player_id"])
        mapping = teams.get(player_id, {})
        rows.append(
            {
                "run_id": run_id,
                "player_id": player_id,
                "game_id": str(source.get("game") or mapping.get("game") or "") or None,
                "player_name": source.get("player") or mapping.get("player") or player_id,
                "position": source.get("position") or mapping.get("position"),
                "team": mapping.get("team") or None,
                "fanduel_mean": source.get("fanduel_mean"),
                "fanduel_p50": source.get("fanduel_p50"),
                "fanduel_p75": source.get("fanduel_p75"),
                "fanduel_p90": source.get("fanduel_p90"),
                "fanduel_p95": source.get("fanduel_p95"),
                "fanduel_p99": source.get("fanduel_p99"),
                "fanduel_15_plus_probability": source.get("fanduel_15_plus_probability"),
                "fanduel_20_plus_probability": source.get("fanduel_20_plus_probability"),
                "fanduel_25_plus_probability": source.get("fanduel_25_plus_probability"),
                "fanduel_30_plus_probability": source.get("fanduel_30_plus_probability"),
                "targets_mean": source.get("targets_mean"),
                "receptions_mean": source.get("receptions_mean"),
                "receiving_yards_mean": source.get("receiving_yards_mean"),
                "receiving_tds_mean": source.get("receiving_tds_mean"),
                "rush_attempts_mean": source.get("rush_attempts_mean"),
                "rushing_yards_mean": source.get("rushing_yards_mean"),
                "rushing_tds_mean": source.get("rushing_tds_mean"),
                "pass_attempts_mean": source.get("pass_attempts_mean"),
                "passing_yards_mean": source.get("passing_yards_mean"),
                "passing_tds_mean": source.get("passing_tds_mean"),
                "interceptions_mean": source.get("interceptions_mean"),
                "payload": {},
            }
        )
    return rows


def _explosive_rows(sim_dir: Path, box_dir: Path, run_id: str) -> list[dict]:
    path = sim_dir / "same_world_player_explosive_summary.csv"
    if not path.exists():
        return []
    frame = pl.read_csv(path)
    teams = _team_map(box_dir)
    skills = _skill_map(sim_dir)
    opponent = _opponent_strength_by_player_family(sim_dir)
    rows = []
    for source in frame.to_dicts():
        player_id = str(source["player_id"])
        family = str(source["play_family"])
        identity = skills.get(player_id, {})
        mapping = teams.get(player_id, {})
        coverage, tackling = opponent.get((player_id, family), (None, None))
        rows.append(
            {
                "run_id": run_id,
                "player_id": player_id,
                "play_family": family,
                "player_name": identity.get("player_name") or mapping.get("player") or player_id,
                "position": identity.get("position") or mapping.get("position"),
                "team": identity.get("team") or mapping.get("team") or None,
                "game_id": mapping.get("game") or None,
                "events": int(source.get("events") or 0),
                "yards_mean": source.get("yards_mean"),
                "touchdowns": int(source.get("touchdowns") or 0),
                "gain_15plus": int(source.get("gain_15plus") or 0),
                "gain_20plus": int(source.get("gain_20plus") or 0),
                "gain_40plus": int(source.get("gain_40plus") or 0),
                "gain_15plus_rate": source.get("gain_15plus_rate"),
                "gain_20plus_rate": source.get("gain_20plus_rate"),
                "gain_40plus_rate": source.get("gain_40plus_rate"),
                "usage_weight": identity.get("usage_weight"),
                "speed_skill": identity.get("speed_skill"),
                "route_separation_skill": identity.get("route_separation_skill"),
                "open_field_skill": identity.get("open_field_skill"),
                "rush_creation_skill": identity.get("rush_creation_skill"),
                "runner_power_skill": identity.get("runner_power_skill"),
                "mobility_skill": identity.get("mobility_skill"),
                "opponent_coverage_mean": coverage,
                "opponent_tackling_mean": tackling,
                "payload": {
                    "runtime_efficiency": _clean(identity.get("efficiency")),
                    "runtime_explosive": _clean(identity.get("explosive")),
                    "evidence_fields": _clean(identity.get("evidence_fields")),
                },
            }
        )
    return rows


def _calibration_rows(sim_dir: Path, run_id: str) -> list[dict]:
    path = sim_dir / "score_anatomy.json"
    if not path.exists():
        return []
    score = json.loads(path.read_text())
    model = score.get("model", {})
    historical = score.get("historical", {})
    rows = []
    for name, model_value in model.items():
        reference = historical.get(name)
        if not isinstance(model_value, (int, float)):
            continue
        absolute_delta = None if reference is None else float(model_value) - float(reference)
        relative_delta = (
            None
            if reference in (None, 0)
            else (float(model_value) - float(reference)) / abs(float(reference))
        )
        rows.append(
            {
                "run_id": run_id,
                "metric_scope": "league",
                "scope_key": "league",
                "metric_name": name,
                "model_value": float(model_value),
                "reference_value": None if reference is None else float(reference),
                "absolute_delta": absolute_delta,
                "relative_delta": relative_delta,
                "reference_season": int(score.get("historical_season", 2025)),
                "details": {},
            }
        )
    return rows


def _input_rows(run_id: str, args: argparse.Namespace) -> list[dict]:
    rows = []
    for name, path, season in (
        ("personnel", args.personnel, 2026),
        ("policy_2025", args.policy, 2025),
        ("player_usage", args.player_usage, 2025),
    ):
        path = Path(path)
        rows.append(
            {
                "run_id": run_id,
                "input_name": name,
                "input_version": path.name,
                "source_kind": "artifact",
                "source_uri": str(path),
                "source_sha256": _sha256(path) if path.exists() and path.is_file() else None,
                "season": season,
                "metadata": {},
            }
        )
    return rows


def _all_output_files(sim_dir: Path, box_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for prefix, root in (("simulation", sim_dir), ("box_scores", box_dir)):
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append((f"{prefix}/{path.relative_to(root).as_posix()}", path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    args = parser.parse_args()

    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        print("MONSTER persistence skipped: no SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY")
        return
    url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
    writer = SupabaseWriter(url, key)

    manifest = json.loads((args.simulation / "simulator_run_manifest.json").read_text())
    github_run_id = os.getenv("GITHUB_RUN_ID", "local")
    git_sha = os.getenv("GITHUB_SHA")
    seed = int(manifest["seed"])
    run_id = os.getenv("MONSTER_RUN_ID") or f"rlv2_{seed}_{github_run_id}"
    parent_run_id = os.getenv("MONSTER_PARENT_RUN_ID") or None
    model_version = os.getenv("MONSTER_MODEL_VERSION", "Reality Loop v2 progressive skill tail")

    run_row = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "model_version": model_version,
        "engine_version": "Monster v1.3 event engine / Reality Loop v2",
        "run_purpose": os.getenv("MONSTER_RUN_PURPOSE", "calibration_experiment"),
        "status": "completed",
        "promotion_status": os.getenv("MONSTER_PROMOTION_STATUS", "shadow"),
        "git_repository": os.getenv("GITHUB_REPOSITORY", "firstrateent-star/nfl-dfs-monster"),
        "git_branch": os.getenv("GITHUB_REF_NAME", "work/reality-loop-v2-test"),
        "git_sha": git_sha,
        "workflow_provider": "github_actions" if github_run_id != "local" else "local",
        "workflow_run_id": None if github_run_id == "local" else int(github_run_id),
        "season": int(manifest["season"]),
        "week": int(manifest["week"]),
        "seed": seed,
        "worlds_per_game": int(manifest["worlds_per_game"]),
        "game_count": int(manifest["games"]),
        "simulated_games": int(manifest["simulated_games"]),
        "player_world_rows": int(manifest["player_world_rows"]),
        "market_blind": True,
        "football_is_source_of_truth": bool(manifest.get("football_is_source_of_truth", True)),
        "fanduel_scoring_downstream_only": bool(
            manifest.get("fanduel_scoring_downstream_only", True)
        ),
        "config": {
            "progressive_skill_tail": True,
            "neutral_historical_center": True,
            "storage_bucket": args.bucket,
        },
        "summary": manifest,
        "completed_at": manifest.get("generated_at_utc"),
        "notes": "Automatic persistence from the same worlds used for scoreboard, stats and DFS.",
    }

    writer.upsert("simulation_runs", [run_row], ("run_id",))
    writer.upsert("run_inputs", _input_rows(run_id, args), ("run_id", "input_name"))
    writer.upsert(
        "game_simulation_summaries",
        _game_rows(args.simulation, run_id),
        ("run_id", "game_id"),
    )
    writer.upsert(
        "player_simulation_summaries",
        _player_rows(args.simulation, args.boxes, run_id),
        ("run_id", "player_id"),
    )
    writer.upsert(
        "player_explosive_summaries",
        _explosive_rows(args.simulation, args.boxes, run_id),
        ("run_id", "player_id", "play_family"),
    )
    writer.upsert(
        "calibration_metrics",
        _calibration_rows(args.simulation, run_id),
        ("run_id", "metric_scope", "scope_key", "metric_name"),
    )

    file_rows = []
    github_url = (
        None
        if github_run_id == "local"
        else f"https://github.com/{os.getenv('GITHUB_REPOSITORY', 'firstrateent-star/nfl-dfs-monster')}/actions/runs/{github_run_id}"
    )
    upload_failures = 0
    for artifact_name, source in _all_output_files(args.simulation, args.boxes):
        storage_path = f"runs/{run_id}/{artifact_name}"
        provider = "supabase_storage"
        bucket = args.bucket
        stored_path = storage_path
        metadata: dict[str, object] = {}
        try:
            writer.upload(args.bucket, storage_path, source)
        except Exception as exc:
            upload_failures += 1
            provider = "github_actions"
            bucket = None
            stored_path = None
            metadata["storage_upload_error"] = str(exc)[:500]
        file_rows.append(
            {
                "run_id": run_id,
                "artifact_name": artifact_name,
                "artifact_type": "simulation_output",
                "storage_provider": provider,
                "bucket": bucket,
                "storage_path": stored_path,
                "external_url": github_url,
                "file_format": source.suffix.lower().lstrip("."),
                "row_count": _row_count(source),
                "byte_size": source.stat().st_size,
                "sha256": _sha256(source),
                "metadata": metadata,
            }
        )
    writer.upsert("simulation_files", file_rows, ("run_id", "artifact_name"))

    print(
        json.dumps(
            {
                "run_id": run_id,
                "games": len(_game_rows(args.simulation, run_id)),
                "players": len(_player_rows(args.simulation, args.boxes, run_id)),
                "player_explosive_rows": len(_explosive_rows(args.simulation, args.boxes, run_id)),
                "calibration_metrics": len(_calibration_rows(args.simulation, run_id)),
                "files": len(file_rows),
                "storage_upload_failures": upload_failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
