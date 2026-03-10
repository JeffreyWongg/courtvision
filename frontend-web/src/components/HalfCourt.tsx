import React from 'react';
import { CANVAS_W, CANVAS_H, SCALE } from './PlayerDot';

/**
 * NBA half-court SVG markings.
 * All measurements are in feet, multiplied by SCALE to get SVG px.
 * Y is flipped: svgY = CANVAS_H - (ft * SCALE), so baseline is at bottom.
 *
 * NBA dimensions used:
 *  - Paint (key): 16 ft wide, 19 ft deep from baseline
 *  - Free-throw circle: 6 ft radius
 *  - Three-point line: 22 ft from basket on sides, arc radius 23.75 ft
 *  - Basket: 5.25 ft from baseline (centre of rim), centred on y=25
 *  - Restricted area arc: 4 ft radius
 */

const S = SCALE;

// Baseline is at x=0 in court coords (left edge of SVG)
// Basket centre in court coords
const BASKET_X_FT = 5.25;
const BASKET_Y_FT = 25; // centre width

// Paint box: 16 ft wide (8 ft each side of centre), 19 ft deep
const PAINT_W_FT = 16;
const PAINT_D_FT = 19;

// Free-throw circle radius
const FT_CIRCLE_R_FT = 6;

// Three-point arc radius from basket centre
const THREE_ARC_R_FT = 23.75;
// Straight portions of the 3pt line extend to y = 25 ± 22 ft (3 ft from sideline)
const THREE_STRAIGHT_Y_OFFSET = 22; // ft from centre

// Restricted area
const RA_R_FT = 4;

// Helper: convert ft to SVG px (Y-flipped)
const px = (xFt: number) => xFt * S;
const py = (yFt: number) => CANVAS_H - yFt * S;

// Build the three-point arc path (right-hand half-court, baseline to left)
function threePointPath(): string {
    const cx = px(BASKET_X_FT);
    const cy = py(BASKET_Y_FT);
    const r = px(THREE_ARC_R_FT);

    // Straight portions stop at y = 25 ± 22
    const yStraightTop = THREE_STRAIGHT_Y_OFFSET;    // 3 ft sideline = y=28
    const yStraightBot = COURT_HEIGHT_FT - THREE_STRAIGHT_Y_OFFSET; // y=22... wait

    // Top straight line endpoints (in SVG coords)
    const topLineStartX = 0; // sideline (baseline edge)
    const topLineStartY = py(BASKET_Y_FT + THREE_STRAIGHT_Y_OFFSET);
    const topLineEndY = topLineStartY;

    // Calculate x where arc meets y = 25 +/- 22
    const dy = (BASKET_Y_FT + THREE_STRAIGHT_Y_OFFSET - BASKET_Y_FT) * S;
    const arcEndX = cx + Math.sqrt(Math.max(0, r * r - dy * dy));

    // Bottom
    const botLineStartY = py(BASKET_Y_FT - THREE_STRAIGHT_Y_OFFSET);
    const botArcEndX = arcEndX;

    return [
        // Top straight portion (from sideline to where arc starts)
        `M ${topLineStartX} ${topLineStartY}`,
        `L ${arcEndX} ${topLineStartY}`,
        // Arc from top to bottom (large arc, sweeping clockwise in SVG)
        `A ${r} ${r} 0 0 0 ${botArcEndX} ${botLineStartY}`,
        // Bottom straight portion back to sideline
        `L ${topLineStartX} ${botLineStartY}`,
    ].join(' ');
}

// Court height in ft
const COURT_HEIGHT_FT = 50;

function restrictedAreaPath(): string {
    const cx = px(BASKET_X_FT);
    const cy = py(BASKET_Y_FT);
    const r = px(RA_R_FT);
    // Semi-circle facing away from baseline (opens toward half-court)
    const startX = cx;
    const startY = cy - r;
    const endX = cx;
    const endY = cy + r;
    return `M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}`;
}

function ftCirclePath(upper: boolean): string {
    const cx = px(BASKET_X_FT + PAINT_D_FT); // at the free-throw line
    const cy = py(BASKET_Y_FT);
    const r = px(FT_CIRCLE_R_FT);
    const startX = cx;
    const startY = cy - r;
    const endX = cx;
    const endY = cy + r;
    // Upper semi-circle (toward half court) or dashed lower half
    if (upper) {
        return `M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}`;
    }
    return `M ${startX} ${startY} A ${r} ${r} 0 0 0 ${endX} ${endY}`;
}

// Lane (free-throw lane) markings — hash marks
function laneHashPaths(): string[] {
    const paths: string[] = [];
    const laneLeft = py(BASKET_Y_FT + PAINT_W_FT / 2);
    const laneRight = py(BASKET_Y_FT - PAINT_W_FT / 2);
    // 4 hash marks on each side at 7, 11, 14, 17 ft from baseline
    for (const dist of [7, 11, 14, 17]) {
        const x = px(dist);
        const hashLen = px(1);
        // Top lane line tick (outward)
        paths.push(`M ${x} ${laneLeft} L ${x} ${laneLeft - hashLen}`);
        // Bottom lane line tick (outward)
        paths.push(`M ${x} ${laneRight} L ${x} ${laneRight + hashLen}`);
    }
    return paths;
}

const HalfCourt: React.FC = () => {
    return (
        <>
            {/* Court background */}
            <rect x={0} y={0} width={CANVAS_W} height={CANVAS_H} fill="#c8902a" rx={4} />

            {/* Court border */}
            <rect
                x={1}
                y={1}
                width={CANVAS_W - 2}
                height={CANVAS_H - 2}
                fill="none"
                stroke="#f5d98b"
                strokeWidth={2}
            />

            {/* Paint / key box */}
            <rect
                x={0}
                y={py(BASKET_Y_FT + PAINT_W_FT / 2)}
                width={px(PAINT_D_FT)}
                height={px(PAINT_W_FT)}
                fill="rgba(180,120,30,0.45)"
                stroke="#f5d98b"
                strokeWidth={1.8}
            />

            {/* Free-throw line */}
            <line
                x1={px(PAINT_D_FT)}
                y1={py(BASKET_Y_FT + PAINT_W_FT / 2)}
                x2={px(PAINT_D_FT)}
                y2={py(BASKET_Y_FT - PAINT_W_FT / 2)}
                stroke="#f5d98b"
                strokeWidth={1.8}
            />

            {/* Free-throw circle — solid upper half */}
            <path
                d={ftCirclePath(true)}
                fill="none"
                stroke="#f5d98b"
                strokeWidth={1.8}
            />
            {/* Free-throw circle — dashed lower half */}
            <path
                d={ftCirclePath(false)}
                fill="none"
                stroke="#f5d98b"
                strokeWidth={1.8}
                strokeDasharray="6 5"
            />

            {/* Lane hash marks */}
            {laneHashPaths().map((d, i) => (
                <path key={i} d={d} stroke="#f5d98b" strokeWidth={1.5} />
            ))}

            {/* Three-point line */}
            <path
                d={threePointPath()}
                fill="none"
                stroke="#f5d98b"
                strokeWidth={1.8}
                strokeLinecap="round"
            />

            {/* Restricted area arc */}
            <path
                d={restrictedAreaPath()}
                fill="none"
                stroke="#f5d98b"
                strokeWidth={1.8}
            />

            {/* Basket backboard */}
            <line
                x1={px(4)}
                y1={py(BASKET_Y_FT + 3)}
                x2={px(4)}
                y2={py(BASKET_Y_FT - 3)}
                stroke="#f5d98b"
                strokeWidth={3}
                strokeLinecap="round"
            />

            {/* Basket rim */}
            <circle
                cx={px(BASKET_X_FT)}
                cy={py(BASKET_Y_FT)}
                r={px(0.75)}
                fill="none"
                stroke="#f97316"
                strokeWidth={2.5}
            />

            {/* Half-court line (right edge of our SVG) */}
            <line
                x1={CANVAS_W}
                y1={0}
                x2={CANVAS_W}
                y2={CANVAS_H}
                stroke="#f5d98b"
                strokeWidth={2}
                strokeDasharray="8 4"
            />
        </>
    );
};

export default HalfCourt;
