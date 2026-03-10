"""POST /api/v1/match — match a player trajectory against the NBA tracking dataset."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_motion_matcher
from app.schemas.player import MatchRequest, MatchResponse
from app.services.motion_matcher import MotionMatcher

router = APIRouter()


@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match trajectory to NBA tracking dataset",
    description=(
        "Accepts an ordered sequence of [x, y] displacement vectors representing "
        "a player trajectory and returns the closest matching play from the NBA "
        "tracking dataset along with a cosine similarity score."
    ),
    tags=["Motion Matching"],
)
def match(
    payload: MatchRequest,
    matcher: MotionMatcher = Depends(get_motion_matcher),
) -> MatchResponse:
    """Find the closest NBA play for the provided trajectory."""
    return matcher.match(payload.trajectory)
