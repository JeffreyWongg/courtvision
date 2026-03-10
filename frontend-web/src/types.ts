/** Coordinate system mirrors the backend Pydantic schemas.
 *  NBA full court: 94 ft (x) × 50 ft (y)
 *  The frontend only renders a HALF court (x: 0–47).
 *  PlayerCoord values sent to the API remain in full-court space.
 */

export interface PlayerCoord {
    x: number; // 0–94 ft (full-court space for API)
    y: number; // 0–50 ft
}

export type Team = 'offense' | 'defense' | 'ball';

/**
 * Placement order for the 11-click flow:
 *   clicks 1–5  → offense players
 *   clicks 6–10 → defense players
 *   click  11   → ball
 */
export type PlacementStep = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 'done';

export interface Player {
    id: string;
    team: Team;
    x: number; // 0–47 ft half-court
    y: number; // 0–50 ft
}

/** POST /api/v1/predict request body */
export interface CourtStateRequest {
    offensive: PlayerCoord[];
    defensive: PlayerCoord[];
}

/** POST /api/v1/predict response */
export interface PredictionResponse {
    offensive: PlayerCoord[];
    defensive: PlayerCoord[];
    model_version: string;
}
