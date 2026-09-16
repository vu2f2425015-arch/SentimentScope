/**
 * SentimentScope Tailwind CSS Configuration
 * Provides dark mode class strategy, custom design system color tokens,
 * typography pairings, and micro-interaction animations.
 */
tailwind.config = {
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                "primary": "var(--color-primary)",
                "primary-hover": "var(--color-primary-hover)",
                "tertiary": "var(--color-tertiary)",
                "surface": "var(--bg-surface)",
                "card": "var(--bg-card)",
                "card-border": "var(--border-card)",
                "background": "var(--bg-app)",
                "on-background": "var(--text-main)",
                "on-surface-variant": "var(--text-muted)",
                "error": "var(--color-error)",
                "success": "var(--color-success)",
                "warning": "var(--color-warning)"
            },
            borderRadius: {
                "DEFAULT": "0.5rem",
                "lg": "0.85rem",
                "xl": "1.25rem",
                "2xl": "1.75rem",
                "3xl": "2.25rem",
                "full": "9999px"
            },
            fontFamily: {
                headline: ["Plus Jakarta Sans", "sans-serif"],
                body: ["Plus Jakarta Sans", "sans-serif"],
                mono: ["JetBrains Mono", "monospace"]
            },
            animation: {
                'pulse-glow': 'pulseGlow 2.5s infinite ease-in-out',
                'float-slow': 'floatSlow 6s ease-in-out infinite',
                'gradient-shift': 'gradientShift 8s ease infinite'
            },
            keyframes: {
                pulseGlow: {
                    '0%, 100%': { opacity: '0.6', filter: 'drop-shadow(0 0 8px rgba(99, 102, 241, 0.4))' },
                    '50%': { opacity: '1', filter: 'drop-shadow(0 0 16px rgba(99, 102, 241, 0.8))' }
                },
                floatSlow: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-6px)' }
                },
                gradientShift: {
                    '0%': { 'background-position': '0% 50%' },
                    '50%': { 'background-position': '100% 50%' },
                    '100%': { 'background-position': '0% 50%' }
                }
            }
        }
    }
};
