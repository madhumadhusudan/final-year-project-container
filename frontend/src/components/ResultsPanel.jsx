import { FileSearchIcon, ShieldIcon } from './Icons.jsx'
import { formatClassName, formatConfidence } from '../utils/detection.js'

function ResultsPanel({ analysisStatus, hasImage, result, error }) {
  const hasResult = analysisStatus === 'success' && result
  const detections = result?.analysis?.detections || []
  const faceDetection = result?.analysis?.face_detection
  const faces = faceDetection?.faces || []
  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="results-title">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-xs font-bold uppercase tracking-[0.16em] text-teal-700">Step 2 of 2</p><h3 id="results-title" className="mt-1 text-lg font-bold text-slate-950">Detection results</h3></div><p className="text-sm text-slate-500">{hasResult ? `Objects: ${result.performance.object_detection_ms} ms · Faces: ${result.performance.face_detection_ms} ms` : analysisStatus === 'analyzing' ? 'Running local detection...' : 'No analysis results yet'}</p></div>
      {analysisStatus === 'error' && <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4" role="alert"><p className="font-bold text-rose-900">Analysis could not be completed</p><p className="mt-1 text-sm text-rose-800">{error}</p></div>}
      {!hasResult && analysisStatus !== 'error' && <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-9 text-center"><FileSearchIcon className="mx-auto h-6 w-6 text-slate-500" /><p className="mt-3 font-bold text-slate-800">{analysisStatus === 'analyzing' ? 'Analyzing image...' : hasImage ? 'Ready to analyze' : 'No image selected'}</p></div>}
      {hasResult && detections.length === 0 && <div className="mt-5 rounded-2xl border border-teal-200 bg-teal-50 px-5 py-7 text-center"><ShieldIcon className="mx-auto h-6 w-6 text-teal-700" /><p className="mt-3 font-bold text-teal-950">No supported objects detected.</p></div>}
      {hasResult && detections.length > 0 && <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{detections.map((detection) => <article key={detection.id} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-bold uppercase text-slate-500">Detection #{detection.id}</p><div className="mt-2 flex items-center justify-between gap-3"><h4 className="font-bold text-slate-950">{formatClassName(detection.class_name)}</h4><span className="text-sm font-bold text-teal-800">{formatConfidence(detection.confidence)}</span></div></article>)}</div>}
      {hasResult && <div className="mt-7 border-t border-slate-200 pt-6"><div className="flex items-center justify-between gap-3"><h3 className="text-lg font-bold text-slate-950">Detected Faces</h3><span className="rounded-full bg-fuchsia-50 px-3 py-1 text-sm font-bold text-fuchsia-800">{faces.length} {faces.length === 1 ? 'Face' : 'Faces'}</span></div>
        {faceDetection?.status === 'error' && <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm font-medium text-amber-900">{faceDetection.error}</p>}
        {faceDetection?.status !== 'error' && faces.length === 0 && <p className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-5 text-center font-semibold text-slate-700">No faces detected.</p>}
        {faces.length > 0 && <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{faces.map((face) => <article key={face.face_id} className="rounded-xl border border-fuchsia-100 bg-fuchsia-50/60 p-4"><p className="font-bold text-slate-950">Face {face.face_id}</p><dl className="mt-2 space-y-1 text-sm"><div className="flex justify-between gap-3"><dt className="text-slate-600">Confidence</dt><dd className="font-bold text-fuchsia-800">{formatConfidence(face.confidence)}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-600">Image coverage</dt><dd className="font-bold text-slate-800">{formatConfidence(face.area_ratio)}</dd></div></dl></article>)}</div>}
      </div>}
    </section>
  )
}

export default ResultsPanel
