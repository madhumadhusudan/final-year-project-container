import { useRef } from 'react'
import ImageInfo from './ImageInfo.jsx'
import ImagePreview from './ImagePreview.jsx'
import ImageUploader from './ImageUploader.jsx'
import { RefreshIcon, SparkIcon, TrashIcon } from './Icons.jsx'

function AnalysisPanel({ selectedFile, previewUrl, imageMetadata, uploadError, analysisStatus, detections, onFileSelect, onRemove, onImageLoaded, onImageError, onAnalyze }) {
  const replacementInputRef = useRef(null)

  function handleReplacement(event) {
    const file = event.target.files?.[0]
    if (file) onFileSelect(file)
    event.target.value = ''
  }

  return (
    <article className="min-w-0 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
      <div className="mb-5 flex items-start justify-between gap-4">
        <div><h3 className="text-lg font-bold text-slate-950">Image analysis</h3><p className="mt-1 text-sm leading-6 text-slate-600">Select one image for local face and person detection.</p></div>
        <span className="shrink-0 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">Step 1 of 2</span>
      </div>
      {!selectedFile ? <ImageUploader onFileSelect={onFileSelect} error={uploadError} /> : (
        <div>
          <ImagePreview previewUrl={previewUrl} filename={selectedFile.name} detections={detections} isAnalyzing={analysisStatus === 'analyzing'} onLoaded={onImageLoaded} onError={onImageError} />
          {uploadError && <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-800" role="alert">{uploadError}</p>}
          {imageMetadata && <ImageInfo metadata={imageMetadata} />}
          <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
            <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700" onClick={() => replacementInputRef.current?.click()}><RefreshIcon className="h-4 w-4" /> Change image</button>
            <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-700 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-600" onClick={onRemove}><TrashIcon className="h-4 w-4" /> Remove image</button>
            <button type="button" disabled={analysisStatus === 'analyzing'} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-teal-700/60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 sm:ml-auto" onClick={onAnalyze}>{analysisStatus === 'analyzing' ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : <SparkIcon className="h-4 w-4" />} {analysisStatus === 'analyzing' ? 'Analyzing…' : 'Analyze Privacy'}</button>
          </div>
          <input ref={replacementInputRef} type="file" className="sr-only" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" aria-label="Choose a replacement image" onChange={handleReplacement} />
          {analysisStatus === 'success' && <div className="mt-5 rounded-xl border border-teal-200 bg-teal-50 px-4 py-3" role="status" aria-live="polite"><p className="font-semibold text-teal-900">Analysis complete.</p><p className="mt-1 text-sm leading-6 text-teal-800">Bounding boxes are mapped from the model’s original-image coordinates.</p></div>}
        </div>
      )}
    </article>
  )
}

export default AnalysisPanel
