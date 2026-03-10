"""Dependency providers for FastAPI routes.

Uses ``functools.lru_cache`` to create singletons — each service is instantiated
once per process lifetime and reused across all requests.

To override in tests, use FastAPI's ``app.dependency_overrides``:

    from app.api.dependencies import get_prediction_service
    app.dependency_overrides[get_prediction_service] = lambda: MockPredictionService()
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.services.motion_matcher import MotionMatcher
from app.services.prediction_service import PredictionService

# Read optional paths from environment variables — makes containerisation easy.
_MODEL_PATH = os.getenv("COURTVISION_MODEL_PATH")
_DATASET_PATH = os.getenv("COURTVISION_DATASET_PATH")


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Singleton ``PredictionService`` instance."""
    return PredictionService(model_path=_MODEL_PATH)


@lru_cache(maxsize=1)
def get_motion_matcher() -> MotionMatcher:
    """Singleton ``MotionMatcher`` instance."""
    return MotionMatcher(dataset_path=_DATASET_PATH)
