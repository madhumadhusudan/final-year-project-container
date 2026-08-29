import { CheckIcon, EyeOffIcon } from './Icons.jsx'

const labels = {
  background_faces: ['background face', 'background faces'],
  license_plates: ['license plate', 'license plates'],
  cards: ['payment card', 'payment cards'],
  sensitive_text: ['sensitive text region', 'sensitive text regions'],
}

function BeforeAfterPanel({ originalUrl, protectedUrl, filename, protection, onDownload }) {
  if (!protectedUrl || !protection) return null
  return (
    <section className="mt-6 rounded-3xl border border-teal-200 bg-white p-4 shadow-sm sm:p-6" aria-labelledby="comparison-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div><p className="text-sm font-semibold uppercase tracking-[0.16em] text-teal-700">Protection complete</p><h3 id="comparison-title" className="mt-1 text-xl font-bold text-slate-950">Before / After comparison</h3><p className="mt-1 text-sm text-slate-600">{protection.regions_protected} privacy {protection.regions_protected === 1 ? 'region' : 'regions'} protected with {protection.method}{protection.method !== 'blackout' ? ` · ${protection.strength}` : ''}.</p></div>
        <button type="button" onClick={onDownload} className="inline-flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 py-2.5 text-sm font-bold text-white"><EyeOffIcon className="h-4 w-4" /> Download Protected Image</button>
      </div>
      {protection.warnings?.map((warning) => <p key={warning} className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm font-semibold text-amber-900">{warning}</p>)}
      {protection.risk && <div className="mt-5 grid gap-3 sm:grid-cols-3" aria-label="Risk reduction">
        <RiskSummary label="Before Protection" value={protection.risk.before} />
        <RiskSummary label="After Protection" value={protection.risk.after} />
        <div className="rounded-2xl border border-teal-200 bg-teal-50 p-4"><p className="text-xs font-bold uppercase tracking-wide text-teal-800">Risk Reduced By</p><p className="mt-2 text-2xl font-black text-teal-950">{protection.risk.reduction} points</p><p className="mt-1 text-sm font-semibold text-teal-800">{protection.risk.reduction_percent}% reduction</p></div>
      </div>}
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <figure className="min-w-0"><figcaption className="mb-2 text-sm font-bold text-slate-700">Original</figcaption><div className="grid min-h-64 place-items-center overflow-hidden rounded-2xl bg-slate-100 ring-1 ring-slate-200"><img src={originalUrl} alt={`Original ${filename}`} className="block max-h-[36rem] max-w-full object-contain" /></div></figure>
        <figure className="min-w-0"><figcaption className="mb-2 text-sm font-bold text-teal-800">Protected</figcaption><div className="grid min-h-64 place-items-center overflow-hidden rounded-2xl bg-slate-100 ring-1 ring-teal-200"><img src={protectedUrl} alt={`Privacy-protected ${filename}`} className="block max-h-[36rem] max-w-full object-contain" /></div></figure>
      </div>
      <div className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {Object.entries(labels).map(([key, [singular, plural]]) => { const count = protection.breakdown?.[key] || 0; return <div key={key} className="rounded-xl bg-slate-50 p-3 text-sm"><span className="font-bold text-slate-950">{count}</span> <span className="text-slate-600">{count === 1 ? singular : plural}</span></div> })}
        <div className={`flex items-center gap-2 rounded-xl p-3 text-sm font-semibold ${protection.main_subject_preserved ? 'bg-teal-50 text-teal-900' : 'bg-slate-50 text-slate-600'}`}><CheckIcon className="h-4 w-4" />{protection.main_subject_preserved ? 'Main subject preserved' : 'No confirmed main subject'}</div>
      </div>
    </section>
  )
}

function RiskSummary({ label, value }) {
  return <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-bold uppercase tracking-wide text-slate-600">{label}</p><p className="mt-2 text-2xl font-black text-slate-950">{value.score} <span className="text-base text-slate-500">/ 100</span></p><p className="mt-1 text-sm font-bold text-slate-700">{value.level}</p></div>
}

export default BeforeAfterPanel
