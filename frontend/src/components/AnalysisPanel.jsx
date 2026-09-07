import { useRef, useState } from 'react'
import ImageInfo from './ImageInfo.jsx'
import ImagePreview from './ImagePreview.jsx'
import ImageUploader from './ImageUploader.jsx'
import { RefreshIcon, SparkIcon, TrashIcon } from './Icons.jsx'

function AnalysisPanel({ selectedFile, previewUrl, imageMetadata, uploadError, analysisStatus, analysisStage, result, onFileSelect, onRemove, onImageLoaded, onImageError, onAnalyze }) {
  const replacementInputRef = useRef(null)
  const [showObjects, setShowObjects] = useState(true)
  const [showFaces, setShowFaces] = useState(true)
  const [showPlates, setShowPlates] = useState(true)
  const [showCards, setShowCards] = useState(true)
  const [showDocuments, setShowDocuments] = useState(true)
  const [showQrCodes, setShowQrCodes] = useState(true)
  const [showBarcodes, setShowBarcodes] = useState(true)
  const [showSensitiveText, setShowSensitiveText] = useState(true)

  function handleReplacement(event) {
    const file = event.target.files?.[0]
    if (file) onFileSelect(file)
    event.target.value = ''
  }

  const detections = result?.analysis?.detections || []
  const faces = result?.analysis?.face_detection?.faces || []
  const plates = result?.analysis?.license_plate_detection?.plates || []
  const cards = result?.analysis?.card_detection?.cards || []
  const documents = result?.analysis?.document_detection?.documents || []
  const qrCodes = result?.analysis?.qr_detection?.items || []
  const barcodes = result?.analysis?.barcode_detection?.items || []
  const sensitiveTexts = result?.analysis?.sensitive_text?.items || []
  return (
    <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
      <div className="mb-5 flex items-start justify-between gap-4"><div><h3 className="text-lg font-bold text-slate-950">Image analysis</h3><p className="mt-1 text-sm leading-6 text-slate-600">Select one image for local privacy detection.</p></div><span className="shrink-0 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">Step 1 of 3</span></div>
      {!selectedFile ? <ImageUploader onFileSelect={onFileSelect} error={uploadError} /> : <div>
        <ImagePreview previewUrl={previewUrl} filename={selectedFile.name} image={result?.image} detections={detections} faces={faces} plates={plates} cards={cards} documents={documents} qrCodes={qrCodes} barcodes={barcodes} sensitiveTexts={sensitiveTexts} showObjects={showObjects} showFaces={showFaces} showPlates={showPlates} showCards={showCards} showDocuments={showDocuments} showQrCodes={showQrCodes} showBarcodes={showBarcodes} showSensitiveText={showSensitiveText} isAnalyzing={analysisStatus === 'analyzing'} analysisStage={analysisStage} onLoaded={onImageLoaded} onError={onImageError} />
        {(detections.length > 0 || faces.length > 0 || plates.length > 0 || cards.length > 0 || documents.length > 0 || qrCodes.length > 0 || barcodes.length > 0 || sensitiveTexts.length > 0) && <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1">
          {detections.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showObjects} onChange={(event) => setShowObjects(event.target.checked)} className="h-5 w-5 accent-cyan-600" /> Show Objects</label>}
          {faces.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showFaces} onChange={(event) => setShowFaces(event.target.checked)} className="h-5 w-5 accent-fuchsia-600" /> Show Faces</label>}
          {plates.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showPlates} onChange={(event) => setShowPlates(event.target.checked)} className="h-5 w-5 accent-amber-500" /> Show License Plates</label>}
          {cards.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showCards} onChange={(event) => setShowCards(event.target.checked)} className="h-5 w-5 accent-emerald-600" /> Show Cards</label>}
          {documents.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showDocuments} onChange={(event) => setShowDocuments(event.target.checked)} className="h-5 w-5 accent-violet-600" /> Show Identity Documents</label>}
          {qrCodes.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showQrCodes} onChange={(event) => setShowQrCodes(event.target.checked)} className="h-5 w-5 accent-sky-600" /> Show QR Codes</label>}
          {barcodes.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showBarcodes} onChange={(event) => setShowBarcodes(event.target.checked)} className="h-5 w-5 accent-orange-600" /> Show Barcodes</label>}
          {sensitiveTexts.length > 0 && <label className="inline-flex min-h-11 items-center gap-3 text-sm font-semibold text-slate-700"><input type="checkbox" checked={showSensitiveText} onChange={(event) => setShowSensitiveText(event.target.checked)} className="h-5 w-5 accent-rose-600" /> Show Sensitive Text</label>}
        </div>}
        {uploadError && <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-800" role="alert">{uploadError}</p>}
        {imageMetadata && <ImageInfo metadata={imageMetadata} />}
        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
          <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 disabled:opacity-50" onClick={() => replacementInputRef.current?.click()}><RefreshIcon className="h-4 w-4" /> Change image</button>
          <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 disabled:opacity-50" onClick={onRemove}><TrashIcon className="h-4 w-4" /> Remove image</button>
          <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-60 sm:ml-auto" onClick={onAnalyze}>{analysisStatus === 'analyzing' ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : <SparkIcon className="h-4 w-4" />} {analysisStatus === 'analyzing' ? 'Analyzing...' : 'Analyze Privacy'}</button>
        </div>
        <input ref={replacementInputRef} type="file" className="sr-only" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" aria-label="Choose a replacement image" onChange={handleReplacement} />
      </div>}
    </article>
  )
}

export default AnalysisPanel
