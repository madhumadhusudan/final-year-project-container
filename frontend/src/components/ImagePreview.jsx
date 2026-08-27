import { ImageIcon } from './Icons.jsx'
import { formatConfidence, getBoundingBoxStyle } from '../utils/detection.js'

function ImagePreview({ previewUrl, filename, detections = [], isAnalyzing = false, onLoaded, onError }) {
  if (!previewUrl) {
    return <div className="grid min-h-72 place-items-center rounded-2xl bg-slate-100 text-slate-400" role="status"><div className="text-center"><ImageIcon className="mx-auto h-8 w-8" /><p className="mt-2 text-sm font-medium">Preparing image preview…</p></div></div>
  }

  return (
    <div className="flex min-h-72 max-h-[34rem] items-center justify-center overflow-hidden rounded-2xl bg-[linear-gradient(45deg,#f1f5f9_25%,transparent_25%),linear-gradient(-45deg,#f1f5f9_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#f1f5f9_75%),linear-gradient(-45deg,transparent_75%,#f1f5f9_75%)] bg-[length:20px_20px] bg-[position:0_0,0_10px,10px_-10px,-10px_0px] ring-1 ring-slate-200">
      <div className="relative inline-flex max-h-[34rem] max-w-full">
        <img src={previewUrl} alt={`Preview of ${filename}`} className="block max-h-[34rem] max-w-full object-contain" onLoad={(event) => onLoaded({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} onError={onError} />
        {detections.map((detection) => {
          const tone = detection.isMainSubject
            ? 'border-teal-400 bg-teal-400/10'
            : detection.category === 'face' ? 'border-amber-400 bg-amber-400/10' : 'border-violet-400 bg-violet-400/10'
          const labelTone = detection.isMainSubject
            ? 'bg-teal-400 text-teal-950'
            : detection.category === 'face' ? 'bg-amber-400 text-amber-950' : 'bg-violet-400 text-violet-950'
          return <div key={detection.id} className={`pointer-events-none absolute border-2 ${tone}`} style={getBoundingBoxStyle(detection.normalizedBoundingBox)} aria-hidden="true"><span className={`absolute left-0 top-0 max-w-[11rem] -translate-y-px whitespace-nowrap rounded-br-md px-1.5 py-1 text-[9px] font-extrabold uppercase leading-none shadow-sm sm:text-[10px] ${labelTone}`}>{detection.label} · {formatConfidence(detection.confidence)}</span></div>
        })}
        {isAnalyzing && <div className="absolute inset-0 grid place-items-center bg-slate-950/55 px-5 text-center text-white" role="status" aria-live="polite"><div><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-white/30 border-t-white" /><p className="mt-3 text-sm font-bold">Running local AI detection…</p><p className="mt-1 text-xs text-white/75">Checking faces and people</p></div></div>}
      </div>
    </div>
  )
}

export default ImagePreview
