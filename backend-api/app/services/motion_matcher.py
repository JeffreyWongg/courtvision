"""MotionMatcher: compares player trajectory vectors against an NBA tracking dataset.

Design notes
------------
- The dataset is expected to be a JSONL file where each record has:
    {"play_id": "...", "description": "...", "trajectory": [[x, y], ...]}
- Similarity is computed via cosine similarity on flattened, zero-padded vectors.
- When the dataset is unavailable a tiny in-memory stub dataset is used so the
  API stays functional without a real sports tracking file.
- The class intentionally exposes ``_compute_similarity`` as a non-private helper
  (single-underscore) so it can be tested directly.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional

from app.schemas.player import MatchResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub dataset — used when no file is provided
# ---------------------------------------------------------------------------
_STUB_DATASET = [
    {
        "play_id": "stub-001",
        "description": "Pick-and-roll drive to the basket",
        "trajectory": [[1.0, 0.5], [2.1, 0.4], [3.0, 0.2], [4.2, -0.1]],
    },
    {
        "play_id": "stub-002",
        "description": "Kick-out three-pointer off a drive",
        "trajectory": [[-1.0, 2.0], [-0.5, 3.1], [0.2, 4.0], [1.0, 4.5]],
    },
    {
        "play_id": "stub-003",
        "description": "Post-up isolation on the low block",
        "trajectory": [[0.0, -1.0], [0.1, -1.5], [-0.1, -2.0], [0.0, -2.5]],
    },
]


@dataclass
class _DatasetEntry:
    play_id: str
    description: str
    trajectory: List[List[float]]


class MotionMatcher:
    """Matches a query trajectory against a catalogue of NBA tracking plays.

    Parameters
    ----------
    dataset_path:
        Path to a JSONL file containing pre-computed trajectories.  When
        ``None`` or the file is missing the in-memory stub dataset is used.

    Example
    -------
    >>> matcher = MotionMatcher()
    >>> from app.schemas.player import MatchRequest
    >>> req = MatchRequest(trajectory=[[1.0, 0.5], [2.0, 0.4]])
    >>> resp = matcher.match(req.trajectory)
    >>> assert 0.0 <= resp.similarity_score <= 1.0
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self._dataset: List[_DatasetEntry] = self._load_dataset(dataset_path)
        logger.info("MotionMatcher initialised with %d plays.", len(self._dataset))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def match(self, trajectory: List[List[float]]) -> MatchResponse:
        """Return the closest matching play for the given trajectory.

        Parameters
        ----------
        trajectory:
            Query trajectory as a list of [x, y] displacement vectors.

        Returns
        -------
        MatchResponse
            Play id, similarity score ∈ [0, 1], and a human-readable description.
        """
        query_vec = self._flatten(trajectory)

        best_entry: Optional[_DatasetEntry] = None
        best_score = -1.0

        for entry in self._dataset:
            candidate_vec = self._flatten(entry.trajectory)
            score = self._compute_similarity(query_vec, candidate_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None:
            # Shouldn't happen — dataset always has at least the stub entries.
            return MatchResponse(match_id="none", similarity_score=0.0, matched_play="No match found.")

        return MatchResponse(
            match_id=best_entry.play_id,
            similarity_score=round(max(0.0, min(1.0, float(best_score))), 6),
            matched_play=best_entry.description,
        )

    # ------------------------------------------------------------------
    # Similarity helpers (single-underscore → accessible for unit tests)
    # ------------------------------------------------------------------

    def _compute_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two equal-length flat vectors.

        Returns a value in [-1, 1]; callers should clamp to [0, 1] for display.
        Zero-vectors return 0.0 by convention.
        """
        if not vec_a or not vec_b:
            return 0.0

        # Zero-pad the shorter vector so they have the same dimension
        length = max(len(vec_a), len(vec_b))
        a = vec_a + [0.0] * (length - len(vec_a))
        b = vec_b + [0.0] * (length - len(vec_b))

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(trajectory: List[List[float]]) -> List[float]:
        """Flatten a list of [x, y] pairs into a 1-D vector."""
        return [coord for pair in trajectory for coord in pair]

    @staticmethod
    def _load_dataset(dataset_path: Optional[str]) -> List[_DatasetEntry]:
        """Load from a JSONL file or fall back to the stub dataset."""
        if dataset_path and os.path.isfile(dataset_path):
            entries: List[_DatasetEntry] = []
            try:
                with open(dataset_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        entries.append(
                            _DatasetEntry(
                                play_id=str(record.get("play_id", "unknown")),
                                description=str(record.get("description", "")),
                                trajectory=record.get("trajectory", []),
                            )
                        )
                logger.info("Loaded %d plays from %r.", len(entries), dataset_path)
                return entries if entries else _MotionMatcher_stub_entries()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load dataset from %r: %s — using stub.", dataset_path, exc)
        else:
            logger.warning("No dataset file found at %r — using stub dataset.", dataset_path)
        return _MotionMatcher_stub_entries()


def _MotionMatcher_stub_entries() -> List[_DatasetEntry]:
    """Convert the module-level stub list into typed entries."""
    return [
        _DatasetEntry(
            play_id=d["play_id"],
            description=d["description"],
            trajectory=d["trajectory"],
        )
        for d in _STUB_DATASET
    ]
