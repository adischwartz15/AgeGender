/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        surface: {
          50: "#f8fafc",
          900: "#0f172a",
        },
      },
    },
  },
  plugins: [],
};
