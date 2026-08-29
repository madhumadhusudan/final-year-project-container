import { FileSearchIcon, ShieldIcon } from './Icons.jsx'
import { formatClassName, formatConfidence } from '../utils/detection.js'
import RiskDashboard from './RiskDashboard.jsx'

function ResultsPanel({ analysisStatus, hasImage, result, error }) {
  const hasResult = analysisStatus === 'success' && result
  const detections = result?.analysis?.detections || []
  const faceDetection = result?.analysis?.face_detection
  const faces = faceDetection?.faces || []
  const mainSubject = result?.analysis?.main_subject
  const plateDetection = result?.analysis?.license_plate_detection
  const cardDetection = result?.analysis?.card_detection
  const plates = plateDetection?.plates || []
  const cards = cardDetection?.cards || []
  const backgroundFaces = faces.filter((face) => face.role === 'background_face')
  const ocr = result?.analysis?.ocr
  const sensitiveText = result?.analysis?.sensitive_text
  const sensitiveItems = sensitiveText?.items || []
  return (
    <section className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-labelledby="results-title">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-xs font-bold uppercase tracking-[0.16em] text-teal-700">Step 2 of 3</p><h3 id="results-title" className="mt-1 text-lg font-bold text-slate-950">Detection results</h3></div><p className="text-sm text-slate-500">{hasResult ? `Objects: ${result.performance.object_detection_ms} ms · Faces: ${result.performance.face_detection_ms} ms` : analysisStatus === 'analyzing' ? 'Running local detection...' : 'No analysis results yet'}</p></div>
      {hasResult && <RiskDashboard risk={result.analysis.privacy_risk} />}
      {analysisStatus === 'error' && <div className="mt-5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-4" role="alert"><p className="font-bold text-rose-900">Analysis could not be completed</p><p className="mt-1 text-sm text-rose-800">{error}</p></div>}
      {!hasResult && analysisStatus !== 'error' && <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-5 py-9 text-center"><FileSearchIcon className="mx-auto h-6 w-6 text-slate-500" /><p className="mt-3 font-bold text-slate-800">{analysisStatus === 'analyzing' ? 'Analyzing image...' : hasImage ? 'Ready to analyze' : 'No image selected'}</p></div>}
      {hasResult && detections.length === 0 && <div className="mt-5 rounded-2xl border border-teal-200 bg-teal-50 px-5 py-7 text-center"><ShieldIcon className="mx-auto h-6 w-6 text-teal-700" /><p className="mt-3 font-bold text-teal-950">No supported objects detected.</p></div>}
      {hasResult && detections.length > 0 && <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{detections.map((detection) => <article key={detection.id} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-bold uppercase text-slate-500">Detection #{detection.id}</p><div className="mt-2 flex items-center justify-between gap-3"><h4 className="font-bold text-slate-950">{formatClassName(detection.class_name)}</h4><span className="text-sm font-bold text-teal-800">{formatConfidence(detection.confidence)}</span></div></article>)}</div>}
      {hasResult && <div className="mt-7 border-t border-slate-200 pt-6"><div className="flex items-center justify-between gap-3"><h3 className="text-lg font-bold text-slate-950">Detected Faces</h3><span className="rounded-full bg-fuchsia-50 px-3 py-1 text-sm font-bold text-fuchsia-800">{faces.length} {faces.length === 1 ? 'Face' : 'Faces'}</span></div>
        {faceDetection?.status === 'error' && <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm font-medium text-amber-900">{faceDetection.error}</p>}
        {faceDetection?.status !== 'error' && faces.length === 0 && <p className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-5 text-center font-semibold text-slate-700">No faces detected.</p>}
        {faces.length > 0 && <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{faces.map((face) => <article key={face.face_id} className="rounded-xl border border-fuchsia-100 bg-fuchsia-50/60 p-4"><p className="font-bold text-slate-950">Face {face.face_id}</p><dl className="mt-2 space-y-1 text-sm"><div className="flex justify-between gap-3"><dt className="text-slate-600">Confidence</dt><dd className="font-bold text-fuchsia-800">{formatConfidence(face.confidence)}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-600">Image coverage</dt><dd className="font-bold text-slate-800">{formatConfidence(face.area_ratio)}</dd></div></dl></article>)}</div>}
      </div>}
      {hasResult && <div className="mt-7 grid gap-5 border-t border-slate-200 pt-6 lg:grid-cols-2">
        <section className="rounded-2xl border border-teal-200 bg-teal-50 p-5"><p className="text-xs font-bold uppercase tracking-[0.14em] text-teal-700">Main Subject</p>
          {mainSubject?.status === 'identified' && <><p className="mt-2 text-lg font-bold text-teal-950">Face {mainSubject.face_id}</p><p className="mt-1 text-sm text-teal-900">Subject confidence: {formatConfidence(mainSubject.subject_score)}</p><p className="mt-2 text-sm text-teal-800">{mainSubject.reason}</p></>}
          {mainSubject?.status === 'uncertain' && <><p className="mt-2 font-bold text-amber-900">Uncertain</p><p className="mt-1 text-sm text-amber-800">{mainSubject.reason}</p></>}
          {mainSubject?.status === 'not_found' && <p className="mt-2 text-sm font-semibold text-slate-700">No main subject found.</p>}
          {backgroundFaces.length > 0 && <p className="mt-3 text-sm font-semibold text-slate-700">Background faces: {backgroundFaces.map((face) => `Face ${face.face_id}`).join(', ')}</p>}
        </section>
        <section className="rounded-2xl border border-slate-200 bg-slate-50 p-5"><p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-600">Privacy-Sensitive Elements</p><dl className="mt-3 grid grid-cols-2 gap-3 text-sm"><div><dt className="text-slate-500">Faces</dt><dd className="text-lg font-bold">{faces.length}</dd></div><div><dt className="text-slate-500">Background Faces</dt><dd className="text-lg font-bold">{backgroundFaces.length}</dd></div><div><dt className="text-slate-500">License Plates</dt><dd className="text-lg font-bold">{plates.length}</dd></div><div><dt className="text-slate-500">Payment Cards</dt><dd className="text-lg font-bold">{cards.length}</dd></div><div><dt className="text-slate-500">Sensitive Text</dt><dd className="text-lg font-bold">{sensitiveItems.length}</dd></div></dl></section>
      </div>}
      {hasResult && <div className="mt-7 grid gap-5 border-t border-slate-200 pt-6 lg:grid-cols-2">
        <PrivacyDetectorSection title="License Plates Detected" noun="License Plate" module={plateDetection} items={plates} />
        <PrivacyDetectorSection title="Payment Cards Detected" noun="Card" module={cardDetection} items={cards} />
      </div>}
      {hasResult && <section className="mt-7 border-t border-slate-200 pt-6"><div className="flex items-center justify-between gap-3"><div><h3 className="text-lg font-bold text-slate-950">Sensitive Text</h3><p className="mt-1 text-sm text-slate-500">{ocr?.text_count || 0} OCR regions checked locally</p></div><span className="rounded-full bg-rose-50 px-3 py-1 text-sm font-bold text-rose-800">{sensitiveItems.length}</span></div>
        {sensitiveText?.status !== 'completed' && <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm font-medium text-amber-900">{sensitiveText?.message || 'OCR is unavailable.'}</p>}
        {sensitiveText?.status === 'completed' && sensitiveItems.length === 0 && <p className="mt-4 rounded-xl bg-slate-50 p-4 text-center text-sm font-semibold text-slate-700">No sensitive text detected.</p>}
        {sensitiveItems.length > 0 && <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{sensitiveItems.map((item) => <article key={item.id} className="rounded-xl border border-rose-100 bg-rose-50/60 p-4"><div className="flex items-start justify-between gap-3"><p className="font-bold text-slate-950">{formatClassName(item.type)}</p><span className="text-sm font-bold text-rose-800">{formatConfidence(item.confidence)}</span></div><p className="mt-2 font-mono text-sm font-bold text-slate-800">{item.masked_value}</p><p className="mt-2 text-xs leading-5 text-slate-600">{item.reason}</p></article>)}</div>}
      </section>}
    </section>
  )
}

function PrivacyDetectorSection({ title, noun, module, items }) {
  return <section><div className="flex items-center justify-between gap-3"><h3 className="font-bold text-slate-950">{title}</h3><span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-bold">{items.length}</span></div>
    {module?.status === 'unavailable' && <p className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm font-medium text-amber-900">{module.message}</p>}
    {module?.status === 'error' && <p className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-medium text-rose-900">{module.message}</p>}
    {module?.status === 'completed' && items.length === 0 && <p className="mt-3 rounded-xl bg-slate-50 p-4 text-sm font-medium text-slate-700">No {title.toLowerCase()}.</p>}
    {items.map((item) => <article key={item.id} className="mt-3 rounded-xl border border-slate-200 p-4"><div className="flex justify-between gap-3"><p className="font-bold">{noun} {item.id}</p><span className="font-bold text-teal-800">{formatConfidence(item.confidence)}</span></div><p className="mt-1 text-sm text-slate-600">{formatClassName(item.class_name)}</p></article>)}
    {module?.diagnostics && <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600"><summary className="cursor-pointer font-bold text-slate-700">Development diagnostics</summary><dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1"><dt>Loaded</dt><dd className="font-semibold">{module.diagnostics.loaded ? 'Yes' : 'No'}</dd><dt>Model</dt><dd className="break-all font-semibold">{module.diagnostics.model_name}</dd><dt>Classes</dt><dd className="break-all font-semibold">{module.diagnostics.model_class_names.join(', ') || 'None'}</dd><dt>Input size</dt><dd className="font-semibold">{module.diagnostics.inference_image_size}px</dd><dt>Threshold</dt><dd className="font-semibold">{module.diagnostics.confidence_threshold}</dd><dt>Tile size / input / count</dt><dd className="font-semibold">{module.diagnostics.tile_size || 'Off'} / {module.diagnostics.tile_inference_image_size || module.diagnostics.inference_image_size}px / {module.diagnostics.tiles_processed}</dd><dt>Raw / accepted</dt><dd className="font-semibold">{module.diagnostics.raw_detection_count} / {module.diagnostics.accepted_detection_count}</dd><dt>Inference</dt><dd className="font-semibold">{module.diagnostics.inference_time_ms} ms</dd></dl></details>}
  </section>
}

export default ResultsPanel
