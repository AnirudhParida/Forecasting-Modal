"""
app/routers/explainability.py
==============================
FastAPI Router for Explainability & Driver Interdependency Analysis Endpoint.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query

from app.models.schemas import ExplainabilityResponse
from app.services.explainability_service import get_explainability_analysis

router = APIRouter(prefix="/api/v1", tags=["Explainability"])


@router.get(
    "/explainability",
    response_model=ExplainabilityResponse,
    summary="Get Explainability & Driver Interdependency Analysis",
    description=(
        "Retrieves forecast quantiles (P50, P5, P95), baseline shift vs historical average, "
        "model confidence, risk level, executive summary, and ranked top interdependency drivers."
    ),
)
async def get_explainability(
    target_metric: Optional[str] = Query(
        "CPU Utilization (%)",
        description="Target metric (e.g. Memory Available (%), CPU Utilization (%))",
    ),
    host_name: str = Query(
        "JPRUPIWEBCRP02",
        description="Host machine name",
    ),
    host_ip: str = Query(
        "10.78.33.83",
        description="Host IP address",
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of top interdependency drivers to return",
    ),
    model: str = Query(
        "STGNN",
        description="Model to use (STGNN or NEURALPROPHET)",
    ),
) -> ExplainabilityResponse:
    """FastAPI endpoint handler for explainability & driver interdependency analysis."""
    return get_explainability_analysis(
        target_metric=target_metric,
        host_name=host_name,
        host_ip=host_ip,
        top_k=top_k,
        model=model,
    )
