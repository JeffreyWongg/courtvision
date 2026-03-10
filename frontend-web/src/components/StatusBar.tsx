import React from 'react';

type Status = 'idle' | 'loading' | 'success' | 'error';

interface StatusBarProps {
    status: Status;
    modelVersion?: string;
    error?: string;
    onReset?: () => void;
}

const StatusBar: React.FC<StatusBarProps> = ({
    status,
    modelVersion,
    error,
    onReset,
}) => {
    const configs: Record<
        Status,
        { bg: string; border: string; icon: string; text: string }
    > = {
        idle: {
            bg: 'rgba(30,30,50,0.7)',
            border: '#4b5563',
            icon: '🏀',
            text: 'Position players on the court, then click Run Predict.',
        },
        loading: {
            bg: 'rgba(30,40,70,0.8)',
            border: '#3b82f6',
            icon: '⏳',
            text: 'Sending court state to model...',
        },
        success: {
            bg: 'rgba(20,50,30,0.8)',
            border: '#22c55e',
            icon: '✅',
            text: `Prediction received — model: ${modelVersion ?? 'unknown'}`,
        },
        error: {
            bg: 'rgba(60,20,20,0.8)',
            border: '#ef4444',
            icon: '⚠️',
            text: error ?? 'Unknown error',
        },
    };

    const cfg = configs[status];

    return (
        <div
            style={{
                background: cfg.bg,
                border: `1px solid ${cfg.border}`,
                borderRadius: 10,
                padding: '10px 18px',
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                fontSize: 13,
                color: '#e5e7eb',
                fontFamily: 'Inter, system-ui, sans-serif',
                backdropFilter: 'blur(8px)',
                transition: 'all 0.3s ease',
                minHeight: 44,
            }}
        >
            <span style={{ fontSize: 18 }}>{cfg.icon}</span>
            <span style={{ flex: 1 }}>{cfg.text}</span>
            {status === 'error' && onReset && (
                <button
                    onClick={onReset}
                    style={{
                        background: 'transparent',
                        border: '1px solid #ef4444',
                        color: '#fca5a5',
                        borderRadius: 6,
                        padding: '3px 10px',
                        cursor: 'pointer',
                        fontSize: 12,
                    }}
                >
                    Dismiss
                </button>
            )}
        </div>
    );
};

export default StatusBar;
