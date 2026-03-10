import type { CourtStateRequest, PredictionResponse } from '../types';

export async function predictCourtState(
    payload: CourtStateRequest,
): Promise<PredictionResponse> {
    const res = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Prediction API error ${res.status}: ${text}`);
    }

    return res.json() as Promise<PredictionResponse>;
}
