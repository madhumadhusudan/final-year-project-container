import { EyeOffIcon, SlidersIcon } from './Icons.jsx'

const riskTypes = ['Background faces', 'Background people']
const plannedTypes = ['License plates', 'Identity documents', 'Sensitive text']
const protectionModes = ['Blur', 'Pixelate', 'Blackout']

function PrivacyControls() {
  return (
    <aside className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="privacy-controls-title">
      <div className="flex items-start gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-slate-100 text-slate-600"><SlidersIcon className="h-5 w-5" /></span>
        <div><h3 id="privacy-controls-title" className="font-bold text-slate-950">Privacy controls</h3><p className="mt-1 text-sm leading-6 text-slate-600">Selective anonymization arrives on Day 4.</p></div>
      </div>
      <fieldset disabled className="mt-6 space-y-2">
        <legend className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Information to protect</legend>
        {riskTypes.map((type) => <label key={type} className="flex cursor-not-allowed items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-500"><span>{type}</span><input type="checkbox" className="h-4 w-4 rounded border-slate-300 accent-teal-700" /></label>)}
      </fieldset>
      <div className="mt-6"><p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Requires custom models</p><div className="mt-3 flex flex-wrap gap-2">{plannedTypes.map((type) => <span key={type} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-500">{type} · planned</span>)}</div></div>
      <fieldset disabled className="mt-6">
        <legend className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Protection method</legend>
        <div className="grid grid-cols-3 gap-2">{protectionModes.map((mode) => <button key={mode} type="button" className="cursor-not-allowed rounded-lg border border-slate-200 bg-slate-50 px-2 py-2 text-xs font-semibold text-slate-400">{mode}</button>)}</div>
      </fieldset>
      <div className="mt-6 flex items-center gap-2 rounded-xl bg-slate-100 px-3 py-3 text-xs font-medium leading-5 text-slate-500"><EyeOffIcon className="h-4 w-4 shrink-0" />Detection recommendations are ready for the Day 4 anonymization engine.</div>
    </aside>
  )
}

export default PrivacyControls
