"""Integration tests for POST /api/v1/match (TDD).

Uses FastAPI TestClient to test the full request/response cycle for the
trajectory-matching endpoint.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestMatchRouteSuccess:
    """Valid trajectory payloads should return HTTP 200 with a MatchResponse."""

    def test_returns_200_for_valid_trajectory(self, client: TestClient, valid_trajectory_payload: dict):
        resp = client.post("/api/v1/match", json=valid_trajectory_payload)
        assert resp.status_code == 200

    def test_response_has_match_id(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert "match_id" in data

    def test_response_has_similarity_score(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert "similarity_score" in data

    def test_response_has_matched_play(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert "matched_play" in data

    def test_similarity_score_is_between_zero_and_one(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert 0.0 <= data["similarity_score"] <= 1.0

    def test_match_id_is_non_empty(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert len(data["match_id"]) > 0

    def test_matched_play_is_non_empty(self, client, valid_trajectory_payload):
        data = client.post("/api/v1/match", json=valid_trajectory_payload).json()
        assert len(data["matched_play"]) > 0

    def test_single_vector_trajectory_accepted(self, client):
        payload = {"trajectory": [[1.0, 0.0]]}
        resp = client.post("/api/v1/match", json=payload)
        assert resp.status_code == 200

    def test_long_trajectory_accepted(self, client):
        payload = {"trajectory": [[float(i), float(i * 0.1)] for i in range(50)]}
        resp = client.post("/api/v1/match", json=payload)
        assert resp.status_code == 200

    def test_negative_displacement_vectors_accepted(self, client):
        """Trajectory vectors represent displacement, not absolute position — negatives are valid."""
        payload = {"trajectory": [[-1.0, -2.0], [-3.0, 0.5]]}
        resp = client.post("/api/v1/match", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Validation errors — expect HTTP 422
# ---------------------------------------------------------------------------

class TestMatchRouteValidationErrors:
    """Invalid trajectory payloads must return HTTP 422."""

    def test_rejects_empty_trajectory(self, client):
        payload = {"trajectory": []}
        assert client.post("/api/v1/match", json=payload).status_code == 422

    def test_rejects_vector_with_one_element(self, client):
        payload = {"trajectory": [[1.0]]}
        assert client.post("/api/v1/match", json=payload).status_code == 422

    def test_rejects_vector_with_three_elements(self, client):
        payload = {"trajectory": [[1.0, 2.0, 3.0]]}
        assert client.post("/api/v1/match", json=payload).status_code == 422

    def test_rejects_non_numeric_vector_elements(self, client):
        payload = {"trajectory": [["a", "b"]]}
        assert client.post("/api/v1/match", json=payload).status_code == 422

    def test_rejects_missing_trajectory_field(self, client):
        assert client.post("/api/v1/match", json={}).status_code == 422

    def test_rejects_trajectory_as_flat_list(self, client):
        """Trajectory must be a list-of-lists, not a flat list."""
        payload = {"trajectory": [1.0, 2.0, 3.0]}
        assert client.post("/api/v1/match", json=payload).status_code == 422
