from __future__ import annotations
from pathlib import Path
import json
import typer
from monster.registry import FeatureRegistry
from monster.settings import Settings
from monster.db import check

app = typer.Typer(no_args_is_help=True)

@app.command("validate-config")
def validate_config(feature_registry: Path = Path("config/feature_registry.yaml")):
    registry = FeatureRegistry.load(feature_registry)
    typer.echo(f"features={len(registry.specs)} blind_allowed={len(registry.blind_allowed())} forbidden={len(registry.forbidden_upstream())}")
    typer.echo(f"registry_sha256={registry.hash()}")

@app.command("db-check")
def db_check():
    settings = Settings.from_env()
    if not settings.database_url:
        raise typer.BadParameter("DATABASE_URL is not configured")
    typer.echo(json.dumps(check(settings.database_url), indent=2))

@app.command("free-budget")
def free_budget():
    typer.echo("Supabase target: <500MB Postgres, <1GB Storage; keep only canonical + rolling world artifacts")
    typer.echo("GitHub target: <2000 Actions min/month; heavy simulator workflow remains manual/on-demand")

if __name__ == "__main__":
    app()
