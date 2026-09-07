import { EyeOffIcon, SlidersIcon } from './Icons.jsx'

const riskTypes = [
  ['protect_background_faces', 'Protect Background Faces'],
  ['protect_license_plates', 'Protect License Plates'],
  ['protect_cards', 'Protect Payment Cards'],
  ['protect_identity_documents', 'Protect Identity Documents'],
  ['protect_qr_codes', 'Protect QR Codes'],
  ['protect_barcodes', 'Protect Barcodes'],
  ['protect_sensitive_text', 'Protect Sensitive Text'],
]
const protectionModes = [['blur', 'Blur'], ['pixelate', 'Pixelate'], ['blackout', 'Blackout']]
const strengths = [['low', 'Low'], ['medium', 'Medium'], ['high', 'High']]

function PrivacyControls({ settings, onChange, onProtect, canProtect, protectionStatus, error, analysis }) {
  const update = (key, value) => onChange({ ...settings, [key]: value })
  const isProtecting = protectionStatus === 'protecting'
  const unavailable = [
    analysis?.license_plate_detection?.status === 'unavailable' && 'Plate detector unavailable',
    analysis?.card_detection?.status === 'unavailable' && 'Card detector unavailable',
    analysis?.document_detection?.status === 'unavailable' && 'Identity document detector unavailable',
    analysis?.qr_detection?.status === 'unavailable' && 'QR detector unavailable',
    analysis?.barcode_detection?.status === 'unavailable' && 'Barcode detector unavailable',
    analysis?.sensitive_text?.status === 'unavailable' && 'OCR unavailable',
  ].filter(Boolean)
  return (
    <aside className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="privacy-controls-title">
      <div className="flex items-start gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-slate-100 text-slate-600"><SlidersIcon className="h-5 w-5" /></span>
        <div><h3 id="privacy-controls-title" className="font-bold text-slate-950">Privacy controls</h3><p className="mt-1 text-sm leading-6 text-slate-600">Choose which detected regions to protect locally.</p></div>
      </div>
      <fieldset disabled={isProtecting} className="mt-6 space-y-2">
        <legend className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Information to protect</legend>
        {riskTypes.map(([key, label]) => {
          const status = key === 'protect_background_faces' ? analysis?.face_detection?.status : key === 'protect_license_plates' ? analysis?.license_plate_detection?.status : key === 'protect_cards' ? analysis?.card_detection?.status : key === 'protect_identity_documents' ? analysis?.document_detection?.status : key === 'protect_qr_codes' ? analysis?.qr_detection?.status : key === 'protect_barcodes' ? analysis?.barcode_detection?.status : analysis?.sensitive_text?.status
          const moduleUnavailable = Boolean(status && status !== 'completed')
          return <label key={key} className={`flex min-h-11 items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium ${moduleUnavailable ? 'cursor-not-allowed text-slate-400' : 'text-slate-700'}`}><span>{label}{moduleUnavailable && <span className="ml-1 text-xs">({status === 'skipped' ? 'not analyzed' : 'unavailable'})</span>}</span><input type="checkbox" disabled={moduleUnavailable} checked={settings[key]} onChange={(event) => update(key, event.target.checked)} className="h-5 w-5 rounded border-slate-300 accent-teal-700" /></label>
        })}
      </fieldset>
      {unavailable.length > 0 && <div className="mt-4 flex flex-wrap gap-2">{unavailable.map((label) => <span key={label} className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-800">{label}</span>)}</div>}
      <fieldset disabled={isProtecting} className="mt-6">
        <legend className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Protection method</legend>
        <div className="grid grid-cols-3 gap-2">{protectionModes.map(([value, label]) => <label key={value} className={`grid min-h-11 place-items-center rounded-lg border px-2 py-2 text-xs font-semibold ${settings.anonymization_method === value ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-slate-200 bg-white text-slate-600'}`}><input type="radio" name="protection-method" value={value} checked={settings.anonymization_method === value} onChange={() => update('anonymization_method', value)} className="sr-only" />{label}</label>)}</div>
      </fieldset>
      <fieldset disabled={isProtecting || settings.anonymization_method === 'blackout'} className="mt-6 disabled:opacity-50">
        <legend className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Strength {settings.anonymization_method === 'blackout' && '· not used for blackout'}</legend>
        <div className="grid grid-cols-3 gap-2">{strengths.map(([value, label]) => <label key={value} className={`grid min-h-11 place-items-center rounded-lg border px-2 py-2 text-xs font-semibold ${settings.strength === value ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-slate-200 bg-white text-slate-600'}`}><input type="radio" name="protection-strength" value={value} checked={settings.strength === value} onChange={() => update('strength', value)} className="sr-only" />{label}</label>)}</div>
      </fieldset>
      <div className="mt-6 flex items-center gap-2 rounded-xl bg-teal-50 px-3 py-3 text-xs font-medium leading-5 text-teal-900"><EyeOffIcon className="h-4 w-4 shrink-0" />A confidently identified main subject is preserved. If identification is uncertain, all detected faces are protected.</div>
      {error && <p className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm font-medium text-rose-900" role="alert">{error}</p>}
      <button type="button" disabled={!canProtect || isProtecting} onClick={onProtect} className="mt-5 inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 py-3 text-sm font-bold text-white shadow-sm disabled:cursor-not-allowed disabled:opacity-50">
        {isProtecting ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : <EyeOffIcon className="h-5 w-5" />}
        {isProtecting ? 'Applying privacy protection...' : 'Protect Image'}
      </button>
      {!canProtect && !isProtecting && <p className="mt-2 text-center text-xs text-slate-500">Upload and analyze an image first.</p>}
    </aside>
  )
}

export default PrivacyControls
