"""Unit tests for MotionMatcher (TDD).

Tests operate entirely on the in-memory stub dataset — no external file required.
"""

from __future__ import annotations

import math

import pytest

from app.schemas.player import MatchResponse
from app.services.motion_matcher import MotionMatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matcher() -> MotionMatcher:
    """Stub MotionMatcher initialised without a dataset file."""
    return MotionMatcher(dataset_path=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMotionMatcherMatchOutput:
    """Structural and type correctness of match() output."""

    def test_returns_match_response(self, matcher):
        result = matcher.match([[1.0, 0.5], [2.0, 0.4]])
        assert isinstance(result, MatchResponse)

    def test_similarity_score_within_zero_one(self, matcher):
        result = matcher.match([[1.0, 0.5], [2.1, 0.4], [3.0, 0.2]])
        assert 0.0 <= result.similarity_score <= 1.0

    def test_match_id_is_non_empty_string(self, matcher):
        result = matcher.match([[0.5, 0.5]])
        assert isinstance(result.match_id, str)
        assert len(result.match_id) > 0

    def test_matched_play_is_non_empty_string(self, matcher):
        result = matcher.match([[0.5, 0.5]])
        assert isinstance(result.matched_play, str)
        assert len(result.matched_play) > 0

    def test_identical_trajectory_returns_high_similarity(self, matcher):
        """Querying the exact same vectors as stub-001 should score very close to 1."""
        stub_001_trajectory = [[1.0, 0.5], [2.1, 0.4], [3.0, 0.2], [4.2, -0.1]]
        # Note: -0.1 displacement is valid; MatchRequest vectors aren't bounded
        result = matcher.match(stub_001_trajectory)
        assert result.similarity_score > 0.98


class TestMotionMatcherComputeSimilarity:
    """_compute_similarity() unit tests — cosine similarity edge cases."""

    def test_identical_vectors_return_one(self, matcher):
        a = [1.0, 2.0, 3.0]
        score = matcher._compute_similarity(a, a)
        assert math.isclose(score, 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors_return_zero(self, matcher):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        score = matcher._compute_similarity(a, b)
        assert math.isclose(score, 0.0, abs_tol=1e-9)

    def test_opposite_vectors_return_negative_one(self, matcher):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        score = matcher._compute_similarity(a, b)
        assert math.isclose(score, -1.0, abs_tol=1e-9)

    def test_zero_vector_a_returns_zero(self, matcher):
        assert matcher._compute_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self, matcher):
        assert matcher._compute_similarity([3.0, 4.0], [0.0, 0.0]) == 0.0

    def test_empty_vectors_return_zero(self, matcher):
        assert matcher._compute_similarity([], []) == 0.0

    def test_different_length_vectors_zero_padded(self, matcher):
        """Shorter vector is zero-padded — result should be valid."""
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0, 0.0]
        score = matcher._compute_similarity(a, b)
        assert math.isclose(score, 1.0, abs_tol=1e-9)

    def test_scaled_vectors_return_same_similarity(self, matcher):
        """Cosine similarity is scale-invariant."""
        a = [1.0, 2.0]
        b = [3.0, 6.0]  # 3× scale of a
        score = matcher._compute_similarity(a, b)
        assert math.isclose(score, 1.0, abs_tol=1e-9)


class TestMotionMatcherFlatten:
    """_flatten() internal helper tests."""

    def test_flattens_two_vectors(self, matcher):
        result = matcher._flatten([[1.0, 2.0], [3.0, 4.0]])
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_flattens_empty_trajectory(self, matcher):
        assert matcher._flatten([]) == []

    def test_flattens_single_vector(self, matcher):
        assert matcher._flatten([[5.5, 6.6]]) == [5.5, 6.6]


class TestMotionMatcherDatasetLoading:
    """Dataset loading fallback behaviour."""

    def test_none_path_uses_stub_dataset(self):
        m = MotionMatcher(dataset_path=None)
        # Stub has 3 entries; at least one match should always be found
        result = m.match([[1.0, 0.5]])
        assert isinstance(result, MatchResponse)

    def test_missing_path_uses_stub_dataset(self, tmp_path):
        m = MotionMatcher(dataset_path=str(tmp_path / "nonexistent.jsonl"))
        result = m.match([[1.0, 0.5]])
        assert isinstance(result, MatchResponse)

    def test_loads_custom_jsonl_dataset(self, tmp_path):
        import json
        dataset_file = tmp_path / "plays.jsonl"
        records = [
            {"play_id": "test-001", "description": "Custom play A", "trajectory": [[1.0, 0.0], [2.0, 0.0]]},
            {"play_id": "test-002", "description": "Custom play B", "trajectory": [[0.0, 1.0], [0.0, 2.0]]},
        ]
        with dataset_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        m = MotionMatcher(dataset_path=str(dataset_file))
        result = m.match([[1.0, 0.0], [2.0, 0.0]])
        assert result.match_id == "test-001"
        assert result.similarity_score > 0.98
