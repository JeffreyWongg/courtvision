"""Pydantic v2 schemas for CourtVision API request/response payloads.

Coordinate system mirrors the existing data-generation CourtState:
  - NBA court: 94 ft (length/x) × 50 ft (width/y)
  - x ∈ [0, 94],  y ∈ [0, 50]
"""

from __future__ import annotations

from typing import Annotated, List

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Court dimension constants (NBA)
# ---------------------------------------------------------------------------
COURT_LENGTH: float = 94.0
COURT_WIDTH: float = 50.0
NUM_PLAYERS: int = 5


# ---------------------------------------------------------------------------
# Coordinate primitive
# ---------------------------------------------------------------------------

class PlayerCoord(BaseModel):
    """A single player's (x, y) court position, validated against NBA bounds."""

    x: Annotated[float, Field(ge=0.0, le=COURT_LENGTH, description="Feet along court length [0, 94]")]
    y: Annotated[float, Field(ge=0.0, le=COURT_WIDTH, description="Feet along court width  [0, 50]")]

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

class CourtStateRequest(BaseModel):
    """10-player court state: exactly 5 offensive and 5 defensive players."""

    offensive: List[PlayerCoord] = Field(
        ...,
        min_length=NUM_PLAYERS,
        max_length=NUM_PLAYERS,
        description="Exactly 5 offensive player coordinates.",
    )
    defensive: List[PlayerCoord] = Field(
        ...,
        min_length=NUM_PLAYERS,
        max_length=NUM_PLAYERS,
        description="Exactly 5 defensive player coordinates.",
    )

    @model_validator(mode="after")
    def _validate_player_counts(self) -> "CourtStateRequest":
        if len(self.offensive) != NUM_PLAYERS:
            raise ValueError(
                f"Exactly {NUM_PLAYERS} offensive players required, got {len(self.offensive)}"
            )
        if len(self.defensive) != NUM_PLAYERS:
            raise ValueError(
                f"Exactly {NUM_PLAYERS} defensive players required, got {len(self.defensive)}"
            )
        return self


class PredictionResponse(BaseModel):
    """Predicted next-frame court state returned by the ML model."""

    offensive: List[PlayerCoord] = Field(..., description="Predicted offensive player positions.")
    defensive: List[PlayerCoord] = Field(..., description="Predicted defensive player positions.")
    model_version: str = Field(..., description="Identifier of the model that produced this prediction.")


# ---------------------------------------------------------------------------
# Match endpoint
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    """A trajectory represented as a sequence of (x, y) displacement vectors."""

    trajectory: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of [x, y] displacement vectors (at least 1 element).",
    )

    @model_validator(mode="after")
    def _validate_vector_shape(self) -> "MatchRequest":
        for i, vec in enumerate(self.trajectory):
            if len(vec) != 2:
                raise ValueError(
                    f"Each trajectory vector must have exactly 2 elements [x, y]; "
                    f"got {len(vec)} at index {i}."
                )
        return self


class MatchResponse(BaseModel):
    """Closest matching play found in the NBA tracking dataset."""

    match_id: str = Field(..., description="Identifier of the matched play in the dataset.")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between query trajectory and matched play [0, 1].",
    )
    matched_play: str = Field(..., description="Human-readable description of the matched play.")
