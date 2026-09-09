from __future__ import annotations

import polars as pl


def _weighted_share(team: pl.DataFrame, mask: pl.Expr, weight: str) -> float:
    total = float(team.select(pl.col(weight).sum()).item()) if team.height else 0.0
    if total <= 0.0:
        return 0.0
    part = float(team.filter(mask).select(pl.col(weight).sum()).item())
    return max(0.0, min(part / total, 1.0))


def _weighted_mean(team: pl.DataFrame, value: str, weight: str) -> float:
    if value not in team.columns or not team.height:
        return 0.0
    valid = team.filter(pl.col(value).is_not_null() & (pl.col(weight) > 0.0))
    if not valid.height:
        return 0.0
    denom = float(valid.select(pl.col(weight).sum()).item())
    if denom <= 0.0:
        return 0.0
    numer = float(valid.select((pl.col(value) * pl.col(weight)).sum()).item())
    return numer / denom


def compile_ol_simulation_context(
    personnel: pl.DataFrame,
    unit_effects: pl.DataFrame,
    historical_outcomes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compile current OL capability, continuity and uncertainty for simulation.

    Current individual blocking capability may affect the mean through the already
    shrunk unit effects. Historical OL outcomes are audit-only here because team policy
    already contains overlapping sack/hit/rushing outcomes. Missing history widens the
    line-state distribution instead of becoming a negative grade.
    """
    required = {
        "team_id",
        "position_group",
        "projected_offense_snap_share",
        "participation_uncertainty",
        "snap_games_observed",
        "prior_team_id",
    }
    missing = required.difference(personnel.columns)
    if missing:
        raise ValueError(f"OL simulation context missing columns: {sorted(missing)}")

    ol = personnel.filter(pl.col("position_group") == "OL")
    rows: list[dict] = []
    for team_id in sorted(personnel.get_column("team_id").unique().to_list()):
        team = ol.filter(pl.col("team_id") == team_id)
        weight = "projected_offense_snap_share"
        total_weight = float(team.select(pl.col(weight).sum()).item()) if team.height else 0.0
        history_coverage = _weighted_share(team, pl.col("snap_games_observed") > 0, weight)
        returning = _weighted_share(
            team,
            (pl.col("snap_games_observed") > 0)
            & pl.col("prior_team_id").is_not_null()
            & (pl.col("prior_team_id") == pl.col("team_id")),
            weight,
        )
        if "madden_pass_block" in team.columns and "madden_run_block" in team.columns:
            madden_coverage = _weighted_share(
                team,
                pl.col("madden_pass_block").is_not_null()
                | pl.col("madden_run_block").is_not_null(),
                weight,
            )
        else:
            madden_coverage = 0.0
        participation_uncertainty = _weighted_mean(team, "participation_uncertainty", weight)

        # The minimum uncertainty is intentionally non-zero: even a fully returning line
        # has week-to-week blocking variance. Missing linkage and turnover widen the tail.
        uncertainty = (
            0.025
            + 0.10 * participation_uncertainty
            + 0.075 * (1.0 - history_coverage)
            + 0.025 * (1.0 - madden_coverage)
            + 0.055 * (1.0 - returning)
        )
        uncertainty = max(0.025, min(uncertainty, 0.18))

        rows.append(
            {
                "team_id": str(team_id),
                "ol_projected_snap_weight": total_weight,
                "ol_history_coverage": history_coverage,
                "ol_returning_snap_share": returning,
                "ol_madden_coverage": madden_coverage,
                "ol_participation_uncertainty": participation_uncertainty,
                "offensive_line_continuity": returning,
                "offensive_line_uncertainty": uncertainty,
                "historical_ol_mean_authority": 0.0,
            }
        )

    context = pl.DataFrame(rows).join(
        unit_effects.select(
            [
                c
                for c in ("team_id", "pass_protection_effect", "run_block_effect")
                if c in unit_effects.columns
            ]
        ),
        on="team_id",
        how="left",
    )

    if historical_outcomes is not None and historical_outcomes.height:
        audit = [
            c
            for c in (
                "team_id",
                "historical_pass_protection_signal",
                "historical_run_block_signal",
                "sack_rate_allowed",
                "qb_hit_rate_allowed",
                "rush_epa_per_play",
                "rush_success_rate",
                "stuff_rate",
                "explosive_rush_rate",
            )
            if c in historical_outcomes.columns
        ]
        context = context.join(historical_outcomes.select(audit), on="team_id", how="left")

    return context.sort("team_id")
