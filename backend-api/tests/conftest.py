"""Pytest fixtures shared across all test modules."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Synchronous HTTPX-backed TestClient wrapping the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Canonical court-state payload (valid, within bounds)
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_court_payload() -> dict:
    """A well-formed 10-player court state matching CourtStateRequest schema."""
    return {
        "offensive": [
            {"x": 60.1, "y": 1.3},
            {"x": 25.9, "y": 11.2},
            {"x": 69.2, "y": 33.8},
            {"x": 83.9, "y": 4.3},
            {"x": 39.7, "y": 1.5},
        ],
        "defensive": [
            {"x": 20.6, "y": 25.3},
            {"x": 2.5,  "y": 9.9},
            {"x": 61.1, "y": 27.2},
            {"x": 20.7, "y": 29.5},
            {"x": 76.1, "y": 0.3},
        ],
    }


@pytest.fixture
def valid_trajectory_payload() -> dict:
    """A valid MatchRequest payload with a two-vector trajectory."""
    return {
        "trajectory": [
            [1.0, 0.5],
            [2.1, 0.4],
            [3.0, 0.2],
        ]
    }
