/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./App.{js,jsx,ts,tsx}", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: '#2A7FFF',
        secondary: '#00C9A7',
        alert: '#FF6B6B',
        surface: '#F7F9FC',
        // Explicit dark mode colors
        'dark-bg': '#0F172A',
        'dark-card': '#1E293B',
        'dark-border': '#334155',
        'dark-text': '#F1F5F9',
        'dark-muted': '#94A3B8',
      },
    },
  },
  plugins: [],
}
