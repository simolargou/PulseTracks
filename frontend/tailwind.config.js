export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b0f0c',
        moss: '#1db954',
        emerald: '#0b5d2b',
        ash: '#9aa4a8'
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif']
      },
      boxShadow: {
        glow: '0 0 30px rgba(29,185,84,0.25)'
      }
    }
  },
  plugins: []
}
