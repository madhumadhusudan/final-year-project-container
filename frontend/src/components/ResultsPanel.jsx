import { FileSearchIcon, ShieldIcon } from './Icons.jsx'
import { formatClassName, formatConfidence } from '../utils/detection.js'

function ResultsPanel({ analysisStatus, hasImage, result, error }) {
  const hasResult = analysisStatus === 'success' && result
  const detections = result?.analysis?.detections || []
  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="results-title">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-xs font-bold uppercase tracking-[0.16em] text-teal-700">Step 2 of 2</p><h3 id="results-title" className="mt-1 text-lg font-bold text-slate-950">Detected objects</h3></div><p className="text-sm text-slate-500">{hasResult ? `YOLOv8n inference: ${result.performance.inference_time_ms} ms` : analysisStatus === 'analyzing' ? 'Running object detection...' : 'No analysis results yet'}</p></div>
      {analysisStatus === 'error' && <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4" role="alert"><p className="font-bold text-rose-900">Analysis could not be completed</p><p className="mt-1 text-sm text-rose-800">{error}</p></div>}
      {!hasResult && analysisStatus !== 'error' && <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-9 text-center"><FileSearchIcon className="mx-auto h-6 w-6 text-slate-500" /><p className="mt-3 font-bold text-slate-800">{analysisStatus === 'analyzing' ? 'Analyzing image...' : hasImage ? 'Ready to analyze' : 'No image selected'}</p></div>}
      {hasResult && detections.length === 0 && <div className="mt-5 rounded-2xl border border-teal-200 bg-teal-50 px-5 py-7 text-center"><ShieldIcon className="mx-auto h-6 w-6 text-teal-700" /><p className="mt-3 font-bold text-teal-950">No supported objects detected.</p></div>}
      {hasResult && detections.length > 0 && <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{detections.map((detection) => <article key={detection.id} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-bold uppercase text-slate-500">Detection #{detection.id}</p><div className="mt-2 flex items-center justify-between gap-3"><h4 className="font-bold text-slate-950">{formatClassName(detection.class_name)}</h4><span className="text-sm font-bold text-teal-800">{formatConfidence(detection.confidence)}</span></div></article>)}</div>}
    </section>
  )
}

export default ResultsPanel
