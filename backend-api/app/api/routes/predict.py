"""POST /api/v1/predict — run ML model inference on a 10-player court state."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_prediction_service
from app.schemas.player import CourtStateRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict next-frame player coordinates",
    description=(
        "Accepts a 10-player court state (5 offensive + 5 defensive) and returns "
        "predicted next-frame coordinates produced by the CourtVision ML model."
    ),
    tags=["Prediction"],
)
def predict(
    payload: CourtStateRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Run model inference and return predicted player positions."""
    return service.predict(payload)
