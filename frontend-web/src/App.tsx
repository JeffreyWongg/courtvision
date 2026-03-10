import React, { useCallback, useState } from 'react';
import { CANVAS_H, CANVAS_W, PlayerDot, svgToFt } from './components/PlayerDot';
import HalfCourt from './components/HalfCourt';
import StatusBar from './components/StatusBar';
import { predictCourtState } from './api/predict';
import type { Player, PlacementStep, PredictionResponse, Team } from './types';

// ─── Placement config ─────────────────────────────────────────────────────────
const TOTAL_CLICKS = 11; // 5 offense + 5 defense + 1 ball

/** What team the nth click should place */
function teamForStep(step: number): Team {
  if (step < 5) return 'offense';
  if (step < 10) return 'defense';
  return 'ball';
}

/** Human-readable instruction for the current click */
function instructionFor(step: PlacementStep): string {
  if (step === 'done') return 'Drag players to adjust positions, then click Run Predict.';
  if (step < 5) return `Click to place Offense ${step + 1} of 5`;
  if (step < 10) return `Click to place Defense ${(step as number) - 4} of 5`;
  return 'Click to place the Ball';
}

/** Accent colour for the current step's instruction banner */
function accentFor(step: PlacementStep): string {
  if (step === 'done') return '#f97316';
  if ((step as number) < 5) return '#ef4444';
  if ((step as number) < 10) return '#3b82f6';
  return '#f59e0b';
}

// ─── App ─────────────────────────────────────────────────────────────────────
type ApiStatus = 'idle' | 'loading' | 'success' | 'error';

const App: React.FC = () => {
  const [players, setPlayers] = useState<Player[]>([]);
  const [placementStep, setPlacementStep] = useState<PlacementStep>(0);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('idle');
  const [modelVersion, setModelVersion] = useState<string | undefined>();
  const [apiError, setApiError] = useState<string | undefined>();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const isPlacing = placementStep !== 'done';

  // ── Derived groups ────────────────────────────────────────────────────────
  const offensePlayers = players.filter(p => p.team === 'offense');
  const defensePlayers = players.filter(p => p.team === 'defense');
  const ball = players.find(p => p.team === 'ball');

  // ── Click-to-place handler ────────────────────────────────────────────────
  const handleCourtClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (placementStep === 'done') return;

      const step = placementStep as number;
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;

      // Scale click coords to the SVG's own coordinate space
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const { x, y } = svgToFt(rawX * scaleX, rawY * scaleY);

      const team = teamForStep(step);
      const teamIndex = team === 'offense' ? step : team === 'defense' ? step - 5 : 0;

      const newPlayer: Player = {
        id: team === 'ball' ? 'ball' : `${team}-${teamIndex}`,
        team,
        x,
        y,
      };

      setPlayers(prev => [...prev, newPlayer]);

      const nextStep = step + 1;
      setPlacementStep(nextStep >= TOTAL_CLICKS ? 'done' : (nextStep as PlacementStep));
    },
    [placementStep],
  );

  // ── Drag handler ──────────────────────────────────────────────────────────
  const handleMove = useCallback((id: string, x: number, y: number) => {
    setPlayers(prev => prev.map(p => (p.id === id ? { ...p, x, y } : p)));
  }, []);

  // ── Predict ───────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    setApiStatus('loading');
    setApiError(undefined);

    const payload = {
      offensive: offensePlayers.map(p => ({ x: p.x * 2, y: p.y })),
      defensive: defensePlayers.map(p => ({ x: p.x * 2, y: p.y })),
    };

    try {
      const response: PredictionResponse = await predictCourtState(payload);
      setPlayers(prev =>
        prev.map(p => {
          if (p.team === 'ball') return p; // ball stays
          if (p.team === 'offense') {
            const idx = offensePlayers.findIndex(o => o.id === p.id);
            const coord = response.offensive[idx];
            return coord ? { ...p, x: coord.x / 2, y: coord.y } : p;
          } else {
            const idx = defensePlayers.findIndex(d => d.id === p.id);
            const coord = response.defensive[idx];
            return coord ? { ...p, x: coord.x / 2, y: coord.y } : p;
          }
        }),
      );
      setModelVersion(response.model_version);
      setApiStatus('success');
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
      setApiStatus('error');
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = () => {
    setPlayers([]);
    setPlacementStep(0);
    setApiStatus('idle');
    setApiError(undefined);
    setModelVersion(undefined);
    setSelectedId(null);
  };

  const selectedPlayer = players.find(p => p.id === selectedId);

  // ── Progress indicators (pills) ───────────────────────────────────────────
  const stepNum = placementStep === 'done' ? TOTAL_CLICKS : (placementStep as number);

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="logo-cluster">
          <span className="logo-icon">🏀</span>
          <div>
            <h1 className="app-title">CourtVision</h1>
            <p className="app-subtitle">Basketball Position Predictor</p>
          </div>
        </div>

        <div className="legend">
          <span className="legend-dot offense" />
          <span className="legend-label">Offense</span>
          <span className="legend-dot defense" />
          <span className="legend-label">Defense</span>
          <span className="legend-dot ball-dot" />
          <span className="legend-label">Ball</span>
        </div>
      </header>

      {/* ── Placement instruction banner ── */}
      <div
        className="placement-banner"
        style={{ borderColor: accentFor(placementStep), background: `${accentFor(placementStep)}14` }}
      >
        <div className="placement-icon">
          {placementStep === 'done' ? '✅' : placementStep < 5 ? '🔴' : placementStep < 10 ? '🔵' : '🏀'}
        </div>
        <div className="placement-text">
          <span className="placement-instruction">{instructionFor(placementStep)}</span>
          {isPlacing && (
            <span className="placement-sub">
              {stepNum} / {TOTAL_CLICKS} placed
            </span>
          )}
        </div>

        {/* Progress pills */}
        <div className="placement-pills">
          {Array.from({ length: TOTAL_CLICKS }).map((_, i) => {
            const team = teamForStep(i);
            const placed = i < stepNum;
            return (
              <span
                key={i}
                className={`pill ${team} ${placed ? 'placed' : ''}`}
              />
            );
          })}
        </div>
      </div>

      {/* ── Main layout ── */}
      <main className="main-layout">
        {/* ── Court canvas ── */}
        <section className="court-section">
          <div className="court-card">
            <svg
              id="half-court-canvas"
              width={CANVAS_W}
              height={CANVAS_H}
              viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
              style={{
                display: 'block',
                borderRadius: 8,
                overflow: 'hidden',
                cursor: isPlacing ? 'crosshair' : 'default',
              }}
              onClick={handleCourtClick}
            >
              <HalfCourt />

              {/* Offense dots */}
              {offensePlayers.map((p, i) => (
                <g
                  key={p.id}
                  onClick={e => {
                    if (!isPlacing) { e.stopPropagation(); setSelectedId(id => (id === p.id ? null : p.id)); }
                  }}
                >
                  <PlayerDot player={p} index={i} onMove={handleMove} disabled={isPlacing || apiStatus === 'loading'} />
                </g>
              ))}

              {/* Defense dots */}
              {defensePlayers.map((p, i) => (
                <g
                  key={p.id}
                  onClick={e => {
                    if (!isPlacing) { e.stopPropagation(); setSelectedId(id => (id === p.id ? null : p.id)); }
                  }}
                >
                  <PlayerDot player={p} index={i} onMove={handleMove} disabled={isPlacing || apiStatus === 'loading'} />
                </g>
              ))}

              {/* Ball */}
              {ball && (
                <g
                  onClick={e => {
                    if (!isPlacing) { e.stopPropagation(); setSelectedId(id => (id === 'ball' ? null : 'ball')); }
                  }}
                >
                  <PlayerDot player={ball} index={0} onMove={handleMove} disabled={isPlacing || apiStatus === 'loading'} />
                </g>
              )}

              {/* Empty-court hint */}
              {players.length === 0 && (
                <text
                  x={CANVAS_W / 2}
                  y={CANVAS_H / 2 + 6}
                  textAnchor="middle"
                  fontSize={13}
                  fill="rgba(245,217,139,0.4)"
                  fontFamily="Inter, system-ui, sans-serif"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  Click the court to place players
                </text>
              )}
            </svg>

            <div className="court-label">
              <span>BASELINE</span>
              <span>HALF COURT →</span>
            </div>
          </div>
        </section>

        {/* ── Side panel ── */}
        <aside className="side-panel">
          {/* Predict button — only enabled when all placed */}
          <button
            id="predict-button"
            className={`predict-btn ${apiStatus === 'loading' ? 'loading' : ''}`}
            onClick={handlePredict}
            disabled={isPlacing || apiStatus === 'loading'}
            title={isPlacing ? 'Place all 11 tokens first' : undefined}
          >
            {apiStatus === 'loading' ? (
              <><span className="spinner" /> Predicting…</>
            ) : (
              <>⚡ Run Predict</>
            )}
          </button>

          <button id="reset-button" className="reset-btn" onClick={handleReset}>
            ↺ Reset Court
          </button>

          {/* Status bar */}
          <StatusBar
            status={apiStatus}
            modelVersion={modelVersion}
            error={apiError}
            onReset={() => setApiStatus('idle')}
          />

          {/* Selected player inspector */}
          <div className="info-card">
            <h3 className="info-title">📍 Player Coordinates</h3>
            {selectedPlayer ? (
              <div className="coord-display">
                <div className="coord-row">
                  <span className="coord-label">Player</span>
                  <span className={`coord-value team-tag ${selectedPlayer.team}`}>
                    {selectedPlayer.team === 'ball' ? 'Ball' : selectedPlayer.team.charAt(0).toUpperCase() + selectedPlayer.team.slice(1)}
                  </span>
                </div>
                <div className="coord-row">
                  <span className="coord-label">X (ft)</span>
                  <span className="coord-value mono">{selectedPlayer.x.toFixed(1)}</span>
                </div>
                <div className="coord-row">
                  <span className="coord-label">Y (ft)</span>
                  <span className="coord-value mono">{selectedPlayer.y.toFixed(1)}</span>
                </div>
              </div>
            ) : (
              <p className="coord-hint">
                {isPlacing
                  ? 'Finish placing all players first.'
                  : 'Click a player dot to inspect its coordinates.'}
              </p>
            )}
          </div>

          {/* Court state list */}
          <div className="info-card players-table-card">
            <h3 className="info-title">🗺 Court State</h3>
            {players.length === 0 ? (
              <p className="coord-hint">No players placed yet.</p>
            ) : (
              <div className="players-table">
                {players.map((p) => {
                  const teamIndex =
                    p.team === 'ball'
                      ? -1
                      : p.team === 'offense'
                        ? offensePlayers.findIndex(d => d.id === p.id)
                        : defensePlayers.findIndex(d => d.id === p.id);
                  const label =
                    p.team === 'ball'
                      ? 'Ball'
                      : `${p.team === 'offense' ? 'O' : 'D'}${teamIndex + 1}`;
                  return (
                    <div
                      key={p.id}
                      className={`player-row ${p.team} ${selectedId === p.id ? 'active' : ''}`}
                      onClick={() => !isPlacing && setSelectedId(id => (id === p.id ? null : p.id))}
                    >
                      <span className={`dot-indicator ${p.team}`} />
                      <span className="player-label">{label}</span>
                      <span className="coord-mono">
                        ({p.x.toFixed(1)}, {p.y.toFixed(1)})
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;
