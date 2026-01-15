import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/cluster', label: 'Clustering Lab' }
]

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-radial text-white">
      <header className="px-6 py-6 md:px-12 flex flex-col md:flex-row md:items-center gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-ash">PulseTracks</p>
          <h1 className="text-3xl md:text-4xl font-display text-glow">Music Insight Studio</h1>
        </div>
        <nav className="flex gap-4 md:ml-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `text-sm uppercase tracking-[0.2em] px-3 py-2 rounded-full transition ${
                  isActive ? 'bg-moss text-ink' : 'text-ash hover:text-white'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="px-6 pb-16 md:px-12">{children}</main>
    </div>
  )
}

export default Layout
