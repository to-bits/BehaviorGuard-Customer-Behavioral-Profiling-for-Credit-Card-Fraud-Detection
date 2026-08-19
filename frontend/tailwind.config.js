/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "sans-serif"],
        display: ["Space Grotesk", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      colors: {
        ink: "#0b1018",
        panel: "#111925",
        line: "#223044",
        signal: "#54c7f3",
        mint: "#6be3b2",
        amber: "#f4bf61",
        danger: "#f57979",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(84, 199, 243, .12), 0 18px 60px rgba(0, 0, 0, .25)",
      },
    },
  },
  plugins: [],
};
