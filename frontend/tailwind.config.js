/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ego: "#facc15",
        aggressive: "#ef4444",
        normal: "#e5e7eb",
        slow: "#3b82f6",
      },
      keyframes: {
        // One-shot flash for the DecisionTicker chip when the action changes.
        // Re-fires by giving the element a React `key` that changes per action.
        flash: {
          "0%": { transform: "scale(1.06)", filter: "brightness(1.6)" },
          "100%": { transform: "scale(1)", filter: "brightness(1)" },
        },
      },
      animation: {
        flash: "flash 500ms ease-out",
      },
    },
  },
  plugins: [],
};
