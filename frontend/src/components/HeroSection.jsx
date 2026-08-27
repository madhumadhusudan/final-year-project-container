import { LockIcon, ShieldIcon } from './Icons.jsx'

function HeroSection() {
  return (
    <section id="home" className="scroll-mt-24 px-4 pb-12 pt-14 sm:px-6 sm:pb-14 sm:pt-18 lg:px-8">
      <div className="mx-auto grid max-w-7xl items-center gap-10 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.15em] text-teal-800"><LockIcon className="h-4 w-4" />Privacy-first image analysis</div>
          <h1 className="mt-5 text-balance text-4xl font-extrabold leading-[1.08] tracking-[-0.035em] text-slate-950 sm:text-5xl lg:text-6xl">Protect Your Privacy <span className="text-teal-700">Before You Share</span></h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600 sm:text-xl">Detect sensitive information in your photos and protect it before posting online.</p>
          <a href="#analyze" className="mt-7 inline-flex items-center justify-center rounded-xl bg-teal-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700">Analyze an image</a>
        </div>
        <div className="hidden rounded-3xl border border-slate-200 bg-white p-6 shadow-sm lg:block" aria-hidden="true">
          <div className="flex items-start gap-4"><span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-teal-100 text-teal-800"><ShieldIcon className="h-6 w-6" /></span><div><p className="font-bold text-slate-900">Share with awareness</p><p className="mt-1 text-sm leading-6 text-slate-600">A deliberate privacy check before every public post.</p></div></div>
          <div className="mt-6 space-y-3 border-t border-slate-100 pt-5">{['Local AI inference', 'No permanent storage', 'You stay in control'].map((item) => <div key={item} className="flex items-center gap-3 text-sm font-medium text-slate-700"><span className="h-2 w-2 rounded-full bg-teal-500" />{item}</div>)}</div>
        </div>
      </div>
    </section>
  )
}

export default HeroSection
