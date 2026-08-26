import { useState } from 'react'
import BackendStatus from './BackendStatus.jsx'
import { CloseIcon, MenuIcon, ShieldIcon } from './Icons.jsx'

const navigation = [
  { label: 'Home', href: '#home' },
  { label: 'Analyze', href: '#analyze' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Privacy', href: '#privacy' },
]

function Navbar({ backendStatus }) {
  const [menuOpen, setMenuOpen] = useState(false)
  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/95 backdrop-blur">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8" aria-label="Main navigation">
        <a href="#home" className="flex min-w-0 items-center gap-2.5 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-teal-700 text-white shadow-sm"><ShieldIcon className="h-5 w-5" /></span>
          <span className="truncate text-sm font-bold tracking-tight text-slate-900 sm:text-base">Social Media Privacy Guard</span>
        </a>
        <div className="hidden items-center gap-7 lg:flex">
          {navigation.map((item) => <a key={item.href} href={item.href} className="text-sm font-medium text-slate-600 transition hover:text-teal-800 focus-visible:rounded focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700">{item.label}</a>)}
        </div>
        <div className="hidden shrink-0 sm:block"><BackendStatus status={backendStatus} /></div>
        <button type="button" className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 lg:hidden" aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'} aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen((open) => !open)}>{menuOpen ? <CloseIcon /> : <MenuIcon />}</button>
      </nav>
      {menuOpen && (
        <div id="mobile-navigation" className="border-t border-slate-200 bg-white px-4 py-4 shadow-lg lg:hidden">
          <div className="mx-auto flex max-w-7xl flex-col gap-1">
            {navigation.map((item) => <a key={item.href} href={item.href} className="rounded-lg px-3 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-teal-700" onClick={() => setMenuOpen(false)}>{item.label}</a>)}
            <div className="mt-3 sm:hidden"><BackendStatus status={backendStatus} /></div>
          </div>
        </div>
      )}
    </header>
  )
}

export default Navbar
