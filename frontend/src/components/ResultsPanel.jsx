import { FileSearchIcon, ImageIcon, ShieldIcon } from './Icons.jsx'

const resultCards = [
  { title: 'Detected privacy risks', description: 'Waiting for analysis', icon: FileSearchIcon },
  { title: 'Privacy risk score', description: 'Available after image analysis', icon: ShieldIcon },
  { title: 'Before / after', description: 'Protected preview will appear here', icon: ImageIcon },
]

function ResultsPanel({ analysisStatus, hasImage }) {
  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="results-title">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-teal-700">Step 2 of 2</p><h3 id="results-title" className="mt-1 text-lg font-bold text-slate-950">Privacy analysis results</h3></div>
        <p className="text-sm text-slate-500">{analysisStatus === 'ready' ? 'Ready for Day 3 AI integration' : hasImage ? 'Run privacy analysis to continue' : 'No analysis results yet'}</p>
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-3">
        {resultCards.map(({ title, description, icon: ResultIcon }) => <div key={title} className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5"><span className="grid h-9 w-9 place-items-center rounded-xl bg-white text-slate-500 ring-1 ring-slate-200"><ResultIcon className="h-5 w-5" /></span><p className="mt-4 text-sm font-bold text-slate-800">{title}</p><p className="mt-1 text-sm leading-6 text-slate-500">{description}</p></div>)}
      </div>
    </section>
  )
}

export default ResultsPanel
