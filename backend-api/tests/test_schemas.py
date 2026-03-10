"""Unit tests for Pydantic schemas (TDD).

Tests are written to drive the exact validation behaviour of PlayerCoord,
CourtStateRequest, PredictionResponse, MatchRequest, and MatchResponse.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.player import (
    CourtStateRequest,
    MatchRequest,
    MatchResponse,
    PlayerCoord,
    PredictionResponse,
)


# ---------------------------------------------------------------------------
# PlayerCoord
# ---------------------------------------------------------------------------

class TestPlayerCoord:
    """Boundary and type tests for the PlayerCoord primitive."""

    def test_valid_coord_within_bounds(self):
        coord = PlayerCoord(x=47.0, y=25.0)
        assert coord.x == 47.0
        assert coord.y == 25.0

    def test_accepts_boundary_values(self):
        PlayerCoord(x=0.0, y=0.0)
        PlayerCoord(x=94.0, y=50.0)

    def test_rejects_x_below_zero(self):
        with pytest.raises(ValidationError, match="x"):
            PlayerCoord(x=-0.1, y=25.0)

    def test_rejects_x_above_court_length(self):
        with pytest.raises(ValidationError, match="x"):
            PlayerCoord(x=94.01, y=25.0)

    def test_rejects_y_below_zero(self):
        with pytest.raises(ValidationError, match="y"):
            PlayerCoord(x=47.0, y=-1.0)

    def test_rejects_y_above_court_width(self):
        with pytest.raises(ValidationError, match="y"):
            PlayerCoord(x=47.0, y=50.01)

    def test_integer_coords_are_coerced_to_float(self):
        coord = PlayerCoord(x=10, y=20)
        assert isinstance(coord.x, float)
        assert isinstance(coord.y, float)


# ---------------------------------------------------------------------------
# CourtStateRequest
# ---------------------------------------------------------------------------

class TestCourtStateRequest:
    """Validation tests for 10-player court state requests."""

    _valid_five = [{"x": 47.0, "y": 25.0}] * 5
    _valid_five_coords = [PlayerCoord(x=47.0, y=25.0)] * 5

    def test_valid_payload_accepted(self):
        req = CourtStateRequest(
            offensive=self._valid_five,
            defensive=self._valid_five,
        )
        assert len(req.offensive) == 5
        assert len(req.defensive) == 5

    def test_rejects_fewer_than_five_offensive(self):
        with pytest.raises(ValidationError):
            CourtStateRequest(
                offensive=[{"x": 47.0, "y": 25.0}] * 4,
                defensive=self._valid_five,
            )

    def test_rejects_more_than_five_offensive(self):
        with pytest.raises(ValidationError):
            CourtStateRequest(
                offensive=[{"x": 47.0, "y": 25.0}] * 6,
                defensive=self._valid_five,
            )

    def test_rejects_fewer_than_five_defensive(self):
        with pytest.raises(ValidationError):
            CourtStateRequest(
                offensive=self._valid_five,
                defensive=[{"x": 47.0, "y": 25.0}] * 4,
            )

    def test_rejects_out_of_bounds_coord_in_offensive(self):
        bad = [{"x": 100.0, "y": 25.0}] + [{"x": 47.0, "y": 25.0}] * 4
        with pytest.raises(ValidationError):
            CourtStateRequest(offensive=bad, defensive=self._valid_five)

    def test_rejects_missing_offensive_field(self):
        with pytest.raises(ValidationError):
            CourtStateRequest(defensive=self._valid_five)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------

class TestPredictionResponse:
    """Serialisation / field tests for PredictionResponse."""

    def test_valid_response_constructed(self):
        coords = [PlayerCoord(x=47.0, y=25.0)] * 5
        resp = PredictionResponse(offensive=coords, defensive=coords, model_version="stub-v0.1.0")
        assert resp.model_version == "stub-v0.1.0"
        assert len(resp.offensive) == 5

    def test_model_version_required(self):
        coords = [PlayerCoord(x=47.0, y=25.0)] * 5
        with pytest.raises(ValidationError):
            PredictionResponse(offensive=coords, defensive=coords)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MatchRequest
# ---------------------------------------------------------------------------

class TestMatchRequest:
    """Validation tests for trajectory match requests."""

    def test_valid_trajectory_accepted(self):
        req = MatchRequest(trajectory=[[1.0, 0.5], [2.0, 0.3]])
        assert len(req.trajectory) == 2

    def test_single_vector_accepted(self):
        req = MatchRequest(trajectory=[[1.0, 0.0]])
        assert len(req.trajectory) == 1

    def test_rejects_empty_trajectory(self):
        with pytest.raises(ValidationError):
            MatchRequest(trajectory=[])

    def test_rejects_vector_with_wrong_dimension(self):
        with pytest.raises(ValidationError):
            MatchRequest(trajectory=[[1.0, 2.0, 3.0]])

    def test_rejects_vector_with_one_element(self):
        with pytest.raises(ValidationError):
            MatchRequest(trajectory=[[1.0]])


# ---------------------------------------------------------------------------
# MatchResponse
# ---------------------------------------------------------------------------

class TestMatchResponse:
    """MatchResponse similarity-score boundary tests."""

    def test_valid_response_at_score_zero(self):
        resp = MatchResponse(match_id="stub-001", similarity_score=0.0, matched_play="desc")
        assert resp.similarity_score == 0.0

    def test_valid_response_at_score_one(self):
        resp = MatchResponse(match_id="stub-001", similarity_score=1.0, matched_play="desc")
        assert resp.similarity_score == 1.0

    def test_rejects_score_above_one(self):
        with pytest.raises(ValidationError):
            MatchResponse(match_id="x", similarity_score=1.1, matched_play="desc")

    def test_rejects_score_below_zero(self):
        with pytest.raises(ValidationError):
            MatchResponse(match_id="x", similarity_score=-0.1, matched_play="desc")
