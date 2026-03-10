"""PredictionService: loads a local ML model and predicts next-frame player coordinates.

Design notes
------------
- The class is initialised with an optional ``model_path``.  When a trained model
  file is available, pass its path here; the service will load it via numpy's
  ``np.load`` (or a custom loader you swap in later).
- When ``model_path`` is ``None`` or the file is missing the service falls back
  to a **stub model** that applies a small bounded random displacement to every
  coordinate.  This keeps the API fully functional end-to-end before a real
  model exists.
- ``predict`` is intentionally synchronous: FastAPI wraps sync route handlers in
  a thread-pool automatically, so there is no need for async here.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from app.schemas.player import (
    COURT_LENGTH,
    COURT_WIDTH,
    CourtStateRequest,
    PlayerCoord,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

_MODEL_VERSION_STUB = "stub-v0.1.0"
_MAX_DISPLACEMENT = 2.0  # feet — maximum random step in stub model


class PredictionService:
    """Loads a basketball position-prediction model and runs inference.

    Parameters
    ----------
    model_path:
        Filesystem path to a saved model artefact.  Pass ``None`` (default) or
        a non-existent path to use the built-in stub.

    Example
    -------
    >>> svc = PredictionService()
    >>> from app.schemas.player import CourtStateRequest, PlayerCoord
    >>> state = CourtStateRequest(
    ...     offensive=[PlayerCoord(x=47, y=25)] * 5,
    ...     defensive=[PlayerCoord(x=50, y=25)] * 5,
    ... )
    >>> resp = svc.predict(state)
    >>> assert len(resp.offensive) == 5
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._model = self._load_model(model_path)
        self._version = _MODEL_VERSION_STUB if self._model is None else self._parse_version(model_path)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        """Identifier string for the currently loaded model."""
        return self._version

    def predict(self, state: CourtStateRequest) -> PredictionResponse:
        """Return predicted next-frame coordinates for all 10 players.

        Parameters
        ----------
        state:
            Validated ``CourtStateRequest`` with 5 offensive and 5 defensive
            player positions.

        Returns
        -------
        PredictionResponse
            Predicted positions clipped to valid court bounds.
        """
        off_arr = self._coords_to_array(state.offensive)
        def_arr = self._coords_to_array(state.defensive)

        pred_off = self._run_model(off_arr)
        pred_def = self._run_model(def_arr)

        return PredictionResponse(
            offensive=self._array_to_coords(pred_off),
            defensive=self._array_to_coords(pred_def),
            model_version=self._version,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: Optional[str]) -> Optional[object]:
        """Attempt to load a model from disk; return ``None`` on failure."""
        if not model_path or not os.path.isfile(model_path):
            logger.warning(
                "No model file found at %r — using stub model.", model_path
            )
            return None
        try:
            model = np.load(model_path, allow_pickle=True)
            logger.info("Model loaded from %r.", model_path)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model: %s — falling back to stub.", exc)
            return None

    @staticmethod
    def _parse_version(model_path: str) -> str:
        """Derive a version string from the model filename."""
        base = os.path.basename(model_path)
        name, _ = os.path.splitext(base)
        return name or "unknown"

    def _run_model(self, coords: np.ndarray) -> np.ndarray:
        """Apply the model (or stub) to a (5, 2) coordinate array."""
        if self._model is None:
            return self._stub_predict(coords)
        # ----------------------------------------------------------------
        # Real model inference goes here.
        # Replace this block with the actual model's forward pass, e.g.:
        #   flat = coords.flatten()
        #   output = self._model.predict(flat.reshape(1, -1))
        #   return output.reshape(5, 2)
        # ----------------------------------------------------------------
        return self._stub_predict(coords)

    @staticmethod
    def _stub_predict(coords: np.ndarray) -> np.ndarray:
        """Stub: add bounded random displacement and clip to court bounds."""
        rng = np.random.default_rng()
        displacement = rng.uniform(-_MAX_DISPLACEMENT, _MAX_DISPLACEMENT, size=coords.shape)
        moved = coords + displacement
        # Clip x to [0, COURT_LENGTH], y to [0, COURT_WIDTH]
        moved[:, 0] = np.clip(moved[:, 0], 0.0, COURT_LENGTH)
        moved[:, 1] = np.clip(moved[:, 1], 0.0, COURT_WIDTH)
        return moved

    @staticmethod
    def _coords_to_array(players: List[PlayerCoord]) -> np.ndarray:
        """Convert a list of ``PlayerCoord`` to a (N, 2) float64 numpy array."""
        return np.array([[p.x, p.y] for p in players], dtype=np.float64)

    @staticmethod
    def _array_to_coords(arr: np.ndarray) -> List[PlayerCoord]:
        """Convert a (N, 2) numpy array back to a list of ``PlayerCoord``."""
        return [PlayerCoord(x=float(row[0]), y=float(row[1])) for row in arr]
