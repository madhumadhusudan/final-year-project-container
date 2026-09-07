const profiles = [
  ['fast', 'Fast', 'Small detector views and targeted OCR'],
  ['balanced', 'Balanced', 'Recommended speed and coverage'],
  ['accuracy', 'Accuracy', 'Larger views and full-frame OCR'],
]

const modules = [
  ['detect_objects', 'Objects'], ['detect_faces', 'Faces'], ['detect_plates', 'Plates'],
  ['detect_cards', 'Cards'], ['detect_documents', 'Documents'], ['detect_qr', 'QR codes'],
  ['detect_barcodes', 'Barcodes'], ['detect_sensitive_text', 'Sensitive text'],
]

function AnalysisPerformanceControls({ options, onChange, disabled }) {
  const update = (key, value) => onChange({ ...options, [key]: value })
  return (
    <div className="mb-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div><h3 className="font-bold text-slate-950">Analysis performance</h3><p className="mt-1 text-sm text-slate-600">Choose the work this analysis should run. Balanced is the default.</p></div>
        <fieldset disabled={disabled} className="grid grid-cols-3 gap-2">
          <legend className="sr-only">Performance profile</legend>
          {profiles.map(([value, label, title]) => <label key={value} title={title} className={`grid min-h-11 cursor-pointer place-items-center rounded-xl border px-3 text-sm font-semibold ${options.performance_profile === value ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-slate-200 text-slate-600'}`}><input className="sr-only" type="radio" name="analysis-profile" checked={options.performance_profile === value} onChange={() => update('performance_profile', value)} />{label}</label>)}
        </fieldset>
      </div>
      <fieldset disabled={disabled} className="mt-4 flex flex-wrap gap-x-5 gap-y-2">
        <legend className="mb-2 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Detectors to run</legend>
        {modules.map(([key, label]) => <label key={key} className="inline-flex min-h-9 items-center gap-2 text-sm font-medium text-slate-700"><input type="checkbox" checked={options[key]} onChange={(event) => update(key, event.target.checked)} className="h-4 w-4 accent-teal-700" />{label}</label>)}
      </fieldset>
    </div>
  )
}

export default AnalysisPerformanceControls
