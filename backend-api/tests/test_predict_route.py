"""Integration tests for POST /api/v1/predict (TDD).

Uses FastAPI TestClient to drive the full request/response cycle, including
Pydantic validation, dependency injection, and JSON serialisation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _five_players(x: float = 47.0, y: float = 25.0) -> list[dict]:
    return [{"x": x, "y": y}] * 5


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPredictRouteSuccess:
    """Valid payloads should return HTTP 200 with a well-formed PredictionResponse."""

    def test_returns_200_for_valid_payload(self, client: TestClient, valid_court_payload: dict):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert resp.status_code == 200

    def test_response_has_offensive_field(self, client, valid_court_payload):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert "offensive" in resp.json()

    def test_response_has_defensive_field(self, client, valid_court_payload):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert "defensive" in resp.json()

    def test_response_has_model_version_field(self, client, valid_court_payload):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert "model_version" in resp.json()

    def test_offensive_has_five_players(self, client, valid_court_payload):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert len(resp.json()["offensive"]) == 5

    def test_defensive_has_five_players(self, client, valid_court_payload):
        resp = client.post("/api/v1/predict", json=valid_court_payload)
        assert len(resp.json()["defensive"]) == 5

    def test_all_returned_coords_have_x_and_y(self, client, valid_court_payload):
        data = client.post("/api/v1/predict", json=valid_court_payload).json()
        for entry in data["offensive"] + data["defensive"]:
            assert "x" in entry
            assert "y" in entry

    def test_all_returned_coords_within_court_bounds(self, client, valid_court_payload):
        data = client.post("/api/v1/predict", json=valid_court_payload).json()
        for entry in data["offensive"] + data["defensive"]:
            assert 0.0 <= entry["x"] <= 94.0
            assert 0.0 <= entry["y"] <= 50.0

    def test_boundary_coordinates_accepted(self, client):
        payload = {
            "offensive": [{"x": 0.0, "y": 0.0}] * 5,
            "defensive": [{"x": 94.0, "y": 50.0}] * 5,
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_model_version_is_non_empty_string(self, client, valid_court_payload):
        data = client.post("/api/v1/predict", json=valid_court_payload).json()
        assert isinstance(data["model_version"], str)
        assert len(data["model_version"]) > 0


# ---------------------------------------------------------------------------
# Validation errors — expect HTTP 422
# ---------------------------------------------------------------------------

class TestPredictRouteValidationErrors:
    """Invalid payloads must return HTTP 422 Unprocessable Entity."""

    def test_rejects_four_offensive_players(self, client):
        payload = {
            "offensive": _five_players()[:4],
            "defensive": _five_players(),
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_six_offensive_players(self, client):
        payload = {
            "offensive": _five_players() + [{"x": 10.0, "y": 10.0}],
            "defensive": _five_players(),
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_four_defensive_players(self, client):
        payload = {
            "offensive": _five_players(),
            "defensive": _five_players()[:4],
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_x_out_of_bounds(self, client):
        payload = {
            "offensive": [{"x": 95.0, "y": 25.0}] + _five_players()[1:],
            "defensive": _five_players(),
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_y_out_of_bounds(self, client):
        payload = {
            "offensive": [{"x": 47.0, "y": -1.0}] + _five_players()[1:],
            "defensive": _five_players(),
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_missing_offensive_field(self, client):
        payload = {"defensive": _five_players()}
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_missing_defensive_field(self, client):
        payload = {"offensive": _five_players()}
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_rejects_empty_body(self, client):
        assert client.post("/api/v1/predict", json={}).status_code == 422

    def test_rejects_non_numeric_coordinate(self, client):
        payload = {
            "offensive": [{"x": "not-a-number", "y": 25.0}] + _five_players()[1:],
            "defensive": _five_players(),
        }
        assert client.post("/api/v1/predict", json=payload).status_code == 422
