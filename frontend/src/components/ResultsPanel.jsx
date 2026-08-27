import { FileSearchIcon, ShieldIcon } from './Icons.jsx'
import { formatCategory, formatConfidence } from '../utils/detection.js'

function SummaryItem({ label, value }) {
  return <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3"><p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</p><p className="mt-1 text-xl font-extrabold text-slate-950">{value}</p></div>
}

function EmptyState({ analysisStatus, hasImage }) {
  const analyzing = analysisStatus === 'analyzing'
  return (
    <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-9 text-center">
      {analyzing ? <span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-teal-200 border-t-teal-700" /> : <span className="mx-auto grid h-10 w-10 place-items-center rounded-xl bg-white text-slate-500 ring-1 ring-slate-200"><FileSearchIcon className="h-5 w-5" /></span>}
      <p className="mt-4 font-bold text-slate-800">{analyzing ? 'Analyzing privacy-sensitive regions' : hasImage ? 'Ready when you are' : 'No image selected'}</p>
      <p className="mx-auto mt-1 max-w-lg text-sm leading-6 text-slate-500">{analyzing ? 'The local model is detecting faces and people, filtering weak predictions, and evaluating the likely main subject.' : hasImage ? 'Select Analyze Privacy to run real local detection.' : 'Choose a JPG, PNG or WEBP image to begin.'}</p>
    </div>
  )
}

function ResultsPanel({ analysisStatus, hasImage, result, error }) {
  const hasResult = analysisStatus === 'success' && result
  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="results-title">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-teal-700">Step 2 of 2</p><h3 id="results-title" className="mt-1 text-lg font-bold text-slate-950">Privacy analysis results</h3></div>
        <p className="text-sm text-slate-500">{hasResult ? `Completed in ${result.processingTimeMs} ms` : analysisStatus === 'analyzing' ? 'AI analysis in progress' : hasImage ? 'Run privacy analysis to continue' : 'No analysis results yet'}</p>
      </div>

      {analysisStatus === 'error' && <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4" role="alert"><p className="font-bold text-rose-900">Analysis could not be completed</p><p className="mt-1 text-sm leading-6 text-rose-800">{error}</p></div>}
      {!hasResult && analysisStatus !== 'error' && <EmptyState analysisStatus={analysisStatus} hasImage={hasImage} />}

      {hasResult && <>
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <SummaryItem label="Detections" value={result.summary.totalObjects} />
          <SummaryItem label="Faces" value={result.summary.faces} />
          <SummaryItem label="People" value={result.summary.people} />
          <SummaryItem label="Main subject" value={result.summary.mainSubjectDetected ? 'Yes' : 'No'} />
          <SummaryItem label="Processing" value={`${result.processingTimeMs} ms`} />
        </div>

        {result.detections.length === 0 ? (
          <div className="mt-5 rounded-2xl border border-teal-200 bg-teal-50 px-5 py-7 text-center"><span className="mx-auto grid h-10 w-10 place-items-center rounded-xl bg-white text-teal-700 ring-1 ring-teal-200"><ShieldIcon className="h-5 w-5" /></span><p className="mt-3 font-bold text-teal-950">No privacy-sensitive objects were detected in this image.</p><p className="mt-1 text-sm leading-6 text-teal-800">No face or person prediction passed the configured confidence thresholds.</p></div>
        ) : (
          <div className="mt-6">
            <h4 className="font-bold text-slate-950">Individual detections</h4>
            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              {result.detections.map((detection, index) => (
                <article key={detection.id} className={`rounded-2xl border p-4 ${detection.isMainSubject ? 'border-teal-200 bg-teal-50/70' : 'border-slate-200 bg-slate-50'}`}>
                  <div className="flex items-start justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.13em] text-slate-500">{formatCategory(detection.category)} #{index + 1}</p><h5 className="mt-1 font-bold text-slate-950">{detection.label}</h5></div><span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-bold ${detection.isMainSubject ? 'bg-teal-700 text-white' : 'bg-slate-200 text-slate-700'}`}>{formatConfidence(detection.confidence)}</span></div>
                  <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-sm"><div><dt className="text-xs text-slate-500">Relative size</dt><dd className="mt-0.5 font-semibold text-slate-800">{(detection.relativeArea * 100).toFixed(1)}%</dd></div><div><dt className="text-xs text-slate-500">Recommendation</dt><dd className="mt-0.5 font-semibold text-slate-800">{detection.recommendedAnonymization ? 'Candidate to anonymize' : 'Protect from auto-blur'}</dd></div></dl>
                  <p className="mt-4 border-t border-slate-200/80 pt-3 text-sm leading-6 text-slate-600">{detection.explanation}</p>
                </article>
              ))}
            </div>
          </div>
        )}
      </>}
    </section>
  )
}

export default ResultsPanel
