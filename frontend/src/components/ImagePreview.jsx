import { ImageIcon } from './Icons.jsx'
import { formatClassName, formatConfidence, getBoundingBoxStyle } from '../utils/detection.js'

function ImagePreview({ previewUrl, filename, image, detections = [], faces = [], plates = [], cards = [], sensitiveTexts = [], showObjects, showFaces, showPlates, showCards, showSensitiveText, isAnalyzing = false, onLoaded, onError }) {
  if (!previewUrl) {
    return <div className="grid min-h-72 place-items-center rounded-2xl bg-slate-100 text-slate-400" role="status"><div className="text-center"><ImageIcon className="mx-auto h-8 w-8" /><p className="mt-2 text-sm font-medium">Preparing image preview...</p></div></div>
  }

  return (
    <div className="flex min-h-72 max-h-[34rem] items-center justify-center overflow-hidden rounded-2xl bg-slate-100 ring-1 ring-slate-200">
      <div className="relative inline-flex max-h-[34rem] max-w-full">
        <img src={previewUrl} alt={`Preview of ${filename}`} className="block max-h-[34rem] max-w-full object-contain" onLoad={(event) => onLoaded({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} onError={onError} />
        {showObjects && detections.map((detection) => (
          <div key={`object-${detection.id}`} className="pointer-events-none absolute border-2 border-cyan-400 bg-cyan-400/10" style={getBoundingBoxStyle(detection.bounding_box, image?.width, image?.height)} aria-hidden="true">
            <span className="absolute left-0 top-0 max-w-[12rem] -translate-y-px whitespace-nowrap rounded-br bg-cyan-400 px-1.5 py-1 text-[10px] font-bold leading-none text-slate-950 shadow-sm">{formatClassName(detection.class_name)} {formatConfidence(detection.confidence)}</span>
          </div>
        ))}
        {showFaces && faces.map((face) => (
          <div key={`face-${face.face_id}`} className="pointer-events-none absolute border-2 border-fuchsia-500 bg-fuchsia-500/10" style={getBoundingBoxStyle(face.bounding_box, image?.width, image?.height)} aria-hidden="true">
            <span className="absolute right-0 top-0 max-w-[12rem] -translate-y-px whitespace-nowrap rounded-bl bg-fuchsia-500 px-1.5 py-1 text-[10px] font-bold leading-none text-white shadow-sm">Face {face.face_id} {formatConfidence(face.confidence)}</span>
          </div>
        ))}
        {showPlates && plates.map((plate) => (
          <div key={`plate-${plate.id}`} className="pointer-events-none absolute border-2 border-amber-400 bg-amber-400/10" style={getBoundingBoxStyle(plate.bounding_box, image?.width, image?.height)} aria-hidden="true">
            <span className="absolute bottom-0 left-0 whitespace-nowrap rounded-tr bg-amber-400 px-1.5 py-1 text-[10px] font-bold leading-none text-slate-950">Plate {plate.id} {formatConfidence(plate.confidence)}</span>
          </div>
        ))}
        {showCards && cards.map((card) => (
          <div key={`card-${card.id}`} className="pointer-events-none absolute border-2 border-emerald-400 bg-emerald-400/10" style={getBoundingBoxStyle(card.bounding_box, image?.width, image?.height)} aria-hidden="true">
            <span className="absolute bottom-0 right-0 whitespace-nowrap rounded-tl bg-emerald-500 px-1.5 py-1 text-[10px] font-bold leading-none text-white">Card {card.id} {formatConfidence(card.confidence)}</span>
          </div>
        ))}
        {showSensitiveText && sensitiveTexts.map((item) => (
          <div key={`sensitive-${item.id}`} className="pointer-events-none absolute border-2 border-rose-500 bg-rose-500/15" style={getBoundingBoxStyle(item.bounding_box, image?.width, image?.height)} aria-hidden="true">
            <span className="absolute left-0 top-0 max-w-[12rem] -translate-y-full whitespace-nowrap rounded-t bg-rose-600 px-1.5 py-1 text-[10px] font-bold leading-none text-white">{formatClassName(item.type)}</span>
          </div>
        ))}
        {isAnalyzing && <div className="absolute inset-0 grid place-items-center bg-slate-950/55 px-5 text-center text-white" role="status" aria-live="polite"><div><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-white/30 border-t-white" /><p className="mt-3 text-sm font-bold">Running privacy detection...</p><p className="mt-1 text-xs text-white/75">Processing objects, faces, and text locally</p></div></div>}
      </div>
    </div>
  )
}

export default ImagePreview
