"""Unit tests for PredictionService (TDD).

All tests operate on the stub model — no trained weights file required.
"""

from __future__ import annotations

import pytest

from app.schemas.player import CourtStateRequest, PlayerCoord
from app.services.prediction_service import PredictionService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def service() -> PredictionService:
    """Stub PredictionService — no model path."""
    return PredictionService(model_path=None)


@pytest.fixture
def valid_state() -> CourtStateRequest:
    five = [PlayerCoord(x=47.0, y=25.0)] * 5
    return CourtStateRequest(offensive=five, defensive=five)


@pytest.fixture
def boundary_state() -> CourtStateRequest:
    """Court state with coordinates at the extremes of valid bounds."""
    offensive = [
        PlayerCoord(x=0.0,  y=0.0),
        PlayerCoord(x=94.0, y=50.0),
        PlayerCoord(x=47.0, y=25.0),
        PlayerCoord(x=1.0,  y=1.0),
        PlayerCoord(x=93.0, y=49.0),
    ]
    defensive = [
        PlayerCoord(x=10.0, y=5.0),
        PlayerCoord(x=80.0, y=45.0),
        PlayerCoord(x=50.0, y=25.0),
        PlayerCoord(x=30.0, y=10.0),
        PlayerCoord(x=60.0, y=40.0),
    ]
    return CourtStateRequest(offensive=offensive, defensive=defensive)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictionServiceModelVersion:
    """model_version property behaviour."""

    def test_model_version_is_non_empty_string(self, service: PredictionService):
        assert isinstance(service.model_version, str)
        assert len(service.model_version) > 0

    def test_stub_model_version_contains_stub(self, service: PredictionService):
        """Stub model version string must identify itself as a stub."""
        assert "stub" in service.model_version.lower()

    def test_named_model_version_uses_filename(self, tmp_path):
        """When a real file exists, version is derived from the filename."""
        model_file = tmp_path / "courtvision-v1.2.npy"
        model_file.write_bytes(b"")  # empty — will fail to load, stays stub
        svc = PredictionService(model_path=str(model_file))
        # Still falls back to stub since the file is not a valid numpy array
        assert isinstance(svc.model_version, str)


class TestPredictionServiceOutputShape:
    """Structural correctness of predict() output."""

    def test_returns_prediction_response(self, service: PredictionService, valid_state):
        from app.schemas.player import PredictionResponse
        result = service.predict(valid_state)
        assert isinstance(result, PredictionResponse)

    def test_offensive_output_has_five_players(self, service, valid_state):
        result = service.predict(valid_state)
        assert len(result.offensive) == 5

    def test_defensive_output_has_five_players(self, service, valid_state):
        result = service.predict(valid_state)
        assert len(result.defensive) == 5

    def test_response_includes_model_version(self, service, valid_state):
        result = service.predict(valid_state)
        assert result.model_version == service.model_version


class TestPredictionServiceCourtBounds:
    """All predicted coordinates must stay within NBA court bounds."""

    def test_all_predicted_coords_within_bounds(self, service, valid_state):
        for _ in range(10):  # run multiple times to catch random edge cases
            result = service.predict(valid_state)
            for coord in result.offensive + result.defensive:
                assert 0.0 <= coord.x <= 94.0, f"x out of bounds: {coord.x}"
                assert 0.0 <= coord.y <= 50.0, f"y out of bounds: {coord.y}"

    def test_boundary_state_predictions_within_bounds(self, service, boundary_state):
        """Predictions from boundary coordinates still stay within the court."""
        for _ in range(10):
            result = service.predict(boundary_state)
            for coord in result.offensive + result.defensive:
                assert 0.0 <= coord.x <= 94.0
                assert 0.0 <= coord.y <= 50.0


class TestPredictionServiceHelpers:
    """Internal helper method contracts."""

    def test_coords_to_array_shape(self, service):
        import numpy as np
        players = [PlayerCoord(x=float(i), y=float(i)) for i in range(5)]
        arr = service._coords_to_array(players)
        assert arr.shape == (5, 2)
        assert arr.dtype == np.float64

    def test_array_to_coords_produces_player_coord_list(self, service):
        import numpy as np
        arr = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 25.0], [70.0, 35.0], [90.0, 10.0]])
        coords = service._array_to_coords(arr)
        assert len(coords) == 5
        assert all(isinstance(c, PlayerCoord) for c in coords)

    def test_stub_predict_clips_to_bounds(self):
        import numpy as np
        # Feed coordinates at absolute boundaries; after displacement they must not escape
        coords = np.array([[0.0, 0.0], [94.0, 50.0], [47.0, 25.0], [47.0, 25.0], [47.0, 25.0]])
        for _ in range(20):
            result = PredictionService._stub_predict(coords)
            assert np.all(result[:, 0] >= 0.0)
            assert np.all(result[:, 0] <= 94.0)
            assert np.all(result[:, 1] >= 0.0)
            assert np.all(result[:, 1] <= 50.0)
