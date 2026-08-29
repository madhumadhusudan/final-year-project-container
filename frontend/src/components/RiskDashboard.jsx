import { formatClassName } from '../utils/detection.js'

const levelStyles = {
  LOW: { text: 'text-emerald-800', bar: 'bg-emerald-500', soft: 'bg-emerald-50 border-emerald-200' },
  MODERATE: { text: 'text-lime-800', bar: 'bg-lime-500', soft: 'bg-lime-50 border-lime-200' },
  ELEVATED: { text: 'text-amber-800', bar: 'bg-amber-500', soft: 'bg-amber-50 border-amber-200' },
  HIGH: { text: 'text-orange-800', bar: 'bg-orange-500', soft: 'bg-orange-50 border-orange-200' },
  CRITICAL: { text: 'text-rose-800', bar: 'bg-rose-600', soft: 'bg-rose-50 border-rose-200' },
}

const breakdownLabels = {
  background_faces: 'Background Faces',
  license_plates: 'License Plates',
  payment_cards: 'Payment Cards',
  sensitive_text: 'Sensitive Text',
  context_uncertainty: 'Context Uncertainty',
}

function RiskDashboard({ risk }) {
  if (!risk) return null
  const styles = levelStyles[risk.level] || levelStyles.MODERATE
  return (
    <section className={`mt-6 rounded-2xl border p-5 sm:p-6 ${styles.soft}`} aria-labelledby="privacy-risk-title">
      <div className="grid gap-6 lg:grid-cols-[minmax(220px,0.75fr)_minmax(0,1.25fr)]">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-600">Local explainable assessment</p>
          <h3 id="privacy-risk-title" className="mt-1 text-xl font-bold text-slate-950">Privacy Risk Score</h3>
          <div className="mt-4 flex items-end gap-2"><span className={`text-5xl font-black tracking-tight ${styles.text}`}>{risk.score}</span><span className="pb-1 text-lg font-bold text-slate-500">/ 100</span></div>
          <p className={`mt-2 text-sm font-black tracking-wide ${styles.text}`}>{risk.level} RISK</p>
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-white/90 ring-1 ring-slate-200" role="progressbar" aria-label="Privacy risk score" aria-valuemin="0" aria-valuemax="100" aria-valuenow={risk.score}>
            <div className={`h-full rounded-full ${styles.bar}`} style={{ width: `${risk.score}%` }} />
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-700">{risk.summary}</p>
        </div>
        <div>
          <h4 className="text-sm font-bold text-slate-950">Risk Breakdown</h4>
          <dl className="mt-3 grid gap-x-5 gap-y-2 sm:grid-cols-2">
            {Object.entries(breakdownLabels).map(([key, label]) => <div key={key} className="flex items-center justify-between gap-3 rounded-lg bg-white/70 px-3 py-2 text-sm"><dt className="text-slate-600">{label}</dt><dd className="font-bold text-slate-950">{risk.breakdown?.[key] || 0}</dd></div>)}
          </dl>
          <div className="mt-3 flex items-center justify-between border-t border-slate-300 pt-3 text-sm"><span className="font-bold text-slate-700">Total Risk</span><span className="font-black text-slate-950">{risk.score}</span></div>
        </div>
      </div>
      {risk.assessment?.status === 'partial' && <div className="mt-5 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950" role="alert"><p className="font-bold">Risk assessment is partial because some detectors are unavailable.</p><p className="mt-1">{risk.assessment.unavailable_modules.map(formatClassName).join(', ')}.</p></div>}
      <div className="mt-5 grid gap-5 border-t border-slate-300 pt-5 md:grid-cols-2">
        <div><h4 className="text-sm font-bold text-slate-950">Top Privacy Risks</h4>{risk.top_risks.length > 0 ? <ul className="mt-3 space-y-2 text-sm text-slate-700">{risk.top_risks.map((item) => <li key={item} className="flex gap-2"><span aria-hidden="true" className="font-bold text-amber-700">!</span><span>{item}</span></li>)}</ul> : <p className="mt-3 text-sm text-slate-600">No detected privacy risk factors.</p>}</div>
        <div><h4 className="text-sm font-bold text-slate-950">Recommended Protection</h4>{risk.recommendations.length > 0 ? <ul className="mt-3 space-y-2 text-sm text-slate-700">{risk.recommendations.map((item) => <li key={item} className="flex gap-2"><span aria-hidden="true" className="font-bold text-teal-700">✓</span><span>{item}</span></li>)}</ul> : <p className="mt-3 text-sm text-slate-600">No protection action is indicated by detected evidence.</p>}</div>
      </div>
    </section>
  )
}

export default RiskDashboard
