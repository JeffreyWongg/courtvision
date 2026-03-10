import React from 'react';
import { motion } from 'framer-motion';
import { useDrag } from '@use-gesture/react';
import type { Player } from '../types';

/** Scale factor: px per foot */
export const SCALE = 9;
/** Half-court length in feet */
export const HALF_COURT_FT = 47;
/** Court width in feet */
export const COURT_WIDTH_FT = 50;
/** SVG canvas dimensions */
export const CANVAS_W = HALF_COURT_FT * SCALE; // 423 px
export const CANVAS_H = COURT_WIDTH_FT * SCALE; // 450 px

/** Convert feet → SVG pixels (Y axis flipped: baseline at bottom) */
export const ftToSvg = (x: number, y: number) => ({
    svgX: x * SCALE,
    svgY: CANVAS_H - y * SCALE,
});

/** Convert SVG pixels → feet */
export const svgToFt = (svgX: number, svgY: number) => ({
    x: Math.max(0, Math.min(HALF_COURT_FT, svgX / SCALE)),
    y: Math.max(0, Math.min(COURT_WIDTH_FT, (CANVAS_H - svgY) / SCALE)),
});

interface PlayerDotProps {
    player: Player;
    /** index within its team group (0-based), used for label */
    index: number;
    onMove: (id: string, x: number, y: number) => void;
    disabled?: boolean;
}

export const PlayerDot: React.FC<PlayerDotProps> = ({
    player,
    index,
    onMove,
    disabled = false,
}) => {
    const { svgX, svgY } = ftToSvg(player.x, player.y);
    const isBall = player.team === 'ball';
    const isOffense = player.team === 'offense';

    const RADIUS = isBall ? 10 : 14;

    // Fill / stroke per team
    const fill = isBall
        ? '#f59e0b'
        : isOffense
            ? '#ef4444'
            : '#3b82f6';

    const stroke = isBall
        ? '#fde68a'
        : isOffense
            ? '#fca5a5'
            : '#93c5fd';

    const glowColor = isBall
        ? 'rgba(245,158,11,0.65)'
        : isOffense
            ? 'rgba(239,68,68,0.55)'
            : 'rgba(59,130,246,0.55)';

    const bindDrag = useDrag(
        ({ offset: [ox, oy], event }) => {
            event.stopPropagation();
            const { x, y } = svgToFt(ox, oy);
            onMove(player.id, x, y);
        },
        {
            bounds: { left: 0, right: CANVAS_W, top: 0, bottom: CANVAS_H },
            from: () => [svgX, svgY],
        },
    );

    return (
        <motion.g
            {...(disabled ? {} : (bindDrag() as object))}
            style={{ cursor: disabled ? 'default' : 'grab', touchAction: 'none' }}
        >
            {/* Animated circle */}
            <motion.circle
                cx={svgX}
                cy={svgY}
                r={RADIUS}
                animate={{ cx: svgX, cy: svgY }}
                transition={{ type: 'spring', stiffness: 180, damping: 22 }}
                fill={fill}
                stroke={stroke}
                strokeWidth={isBall ? 2 : 2.5}
                filter={`drop-shadow(0 2px 7px ${glowColor})`}
            />

            {/* Ball seam lines */}
            {isBall && (
                <>
                    <motion.path
                        animate={{
                            d: `M ${svgX - RADIUS + 3} ${svgY} Q ${svgX} ${svgY - RADIUS * 0.8} ${svgX + RADIUS - 3} ${svgY}`,
                        }}
                        transition={{ type: 'spring', stiffness: 180, damping: 22 }}
                        fill="none"
                        stroke="#92400e"
                        strokeWidth={1.2}
                    />
                    <motion.path
                        animate={{
                            d: `M ${svgX - RADIUS + 3} ${svgY} Q ${svgX} ${svgY + RADIUS * 0.8} ${svgX + RADIUS - 3} ${svgY}`,
                        }}
                        transition={{ type: 'spring', stiffness: 180, damping: 22 }}
                        fill="none"
                        stroke="#92400e"
                        strokeWidth={1.2}
                    />
                    <motion.line
                        animate={{ x1: svgX, y1: svgY - RADIUS, x2: svgX, y2: svgY + RADIUS }}
                        transition={{ type: 'spring', stiffness: 180, damping: 22 }}
                        stroke="#92400e"
                        strokeWidth={1.2}
                    />
                </>
            )}

            {/* Player number (offense / defense only) */}
            {!isBall && (
                <motion.text
                    animate={{ x: svgX, y: svgY + 5 }}
                    transition={{ type: 'spring', stiffness: 180, damping: 22 }}
                    textAnchor="middle"
                    fontSize={11}
                    fontWeight="bold"
                    fill="white"
                    fontFamily="Inter, system-ui, sans-serif"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                    {index + 1}
                </motion.text>
            )}
        </motion.g>
    );
};
