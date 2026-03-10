"""CourtVision FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import match, predict

app = FastAPI(
    title="CourtVision API",
    version="0.1.0",
    description=(
        "Basketball ML inference backend for CourtVision.\n\n"
        "- **POST /api/v1/predict** — run the position-prediction model on a 10-player court state.\n"
        "- **POST /api/v1/match** — find the closest NBA play for a trajectory vector sequence."
    ),
    contact={"name": "CourtVision Team"},
)

# Allow all origins during local development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict.router, prefix="/api/v1")
app.include_router(match.router, prefix="/api/v1")


@app.get("/health", tags=["Meta"], summary="Health check")
def health() -> dict:
    """Return service liveness status."""
    return {"status": "ok", "service": "courtvision-api"}
