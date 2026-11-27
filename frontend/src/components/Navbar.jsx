import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { HeartPulse, Menu, X } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', label: 'Home', end: true },
  { to: '/assess', label: 'Assess' },
  { to: '/performance', label: 'Model Performance' },
  { to: '/about', label: 'About' },
]

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  const linkClass = ({ isActive }) =>
    `px-3.5 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
      isActive
        ? 'bg-teal-500/15 text-teal-300 border border-teal-500/30'
        : 'text-slate-300 hover:text-white hover:bg-slate-800/70 border border-transparent'
    }`

  return (
    <header className="sticky top-0 z-40 border-b border-[#1E3350] bg-[#07111F]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3.5 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-2.5 group">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-500/15 border border-teal-500/30 text-teal-400 transition-transform duration-200 group-hover:scale-105">
            <HeartPulse className="h-5 w-5" />
          </div>
          <div>
            <span className="text-lg font-bold tracking-tight text-white">
              Cardio<span className="text-teal-400">Sense</span>
            </span>
            <p className="hidden text-[11px] text-slate-400 sm:block">
              Explainable 10-Year CHD Risk
            </p>
          </div>
        </NavLink>

        {/* Desktop navigation */}
        <nav className="hidden items-center gap-1.5 md:flex">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={linkClass}>
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* Mobile menu button */}
        <button
          type="button"
          onClick={() => setMobileOpen((prev) => !prev)}
          className="inline-flex items-center justify-center rounded-lg border border-[#1E3350] bg-[#0D1B2E] p-2 text-slate-300 hover:text-white md:hidden"
          aria-label="Toggle navigation menu"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile navigation drawer */}
      {mobileOpen && (
        <nav className="border-t border-[#1E3350] bg-[#0B1728] px-4 py-3 space-y-1.5 md:hidden">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `block rounded-lg px-3.5 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-teal-500/15 text-teal-300 border border-teal-500/30'
                    : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  )
}

