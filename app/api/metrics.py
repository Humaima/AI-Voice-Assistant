"""
Standard Prometheus scrape endpoint. The actual metric definitions
live in app/core/metrics.py (imported wherever they're incremented);
this router just exposes the global registry over HTTP.
"""
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """Deliberately not behind the debug-token guard (app/api/debug_auth.py)
    — a metrics endpoint's contents are aggregate counts, not anything
    that costs money to query, and most Prometheus/Grafana setups
    expect to reach it without custom auth headers. If you want it
    private, put it behind a network-level restriction (e.g. your
    platform's internal networking) rather than an application-level
    token."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
