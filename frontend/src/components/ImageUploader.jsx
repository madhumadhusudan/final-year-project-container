import { useRef, useState } from 'react'
import { UploadIcon } from './Icons.jsx'

function ImageUploader({ onFileSelect, error }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  function chooseFile(file) {
    if (file) onFileSelect(file)
  }

  function handleInputChange(event) {
    chooseFile(event.target.files?.[0])
    event.target.value = ''
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  return (
    <div>
      <button
        type="button"
        className={`group flex min-h-72 w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700 ${isDragging ? 'border-teal-600 bg-teal-50' : 'border-slate-300 bg-slate-50/70 hover:border-teal-500 hover:bg-teal-50/60'}`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => { event.preventDefault(); setIsDragging(true) }}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
        onDragLeave={(event) => {
          event.preventDefault()
          if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false)
        }}
        onDrop={handleDrop}
        aria-describedby="upload-formats upload-privacy upload-error"
      >
        <span className="grid h-14 w-14 place-items-center rounded-2xl bg-white text-teal-700 shadow-sm ring-1 ring-slate-200 transition group-hover:scale-[1.03]"><UploadIcon className="h-7 w-7" /></span>
        <span className="mt-5 text-lg font-bold text-slate-900">Drop your image here</span>
        <span className="mt-1 text-sm text-slate-600">or <span className="font-semibold text-teal-700">click to browse</span></span>
        <span id="upload-formats" className="mt-4 rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-500 ring-1 ring-slate-200">JPG, PNG or WEBP · up to 10 MB</span>
        <span id="upload-privacy" className="mt-4 text-xs leading-5 text-slate-500">Processed temporarily by your local backend. Not stored or sent to a third party.</span>
      </button>
      <input ref={inputRef} type="file" className="sr-only" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" onChange={handleInputChange} aria-label="Choose a JPG, PNG or WEBP image" />
      <div id="upload-error" className="mt-3 min-h-6" role="alert" aria-live="polite">
        {error && <p className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-800">{error}</p>}
      </div>
    </div>
  )
}

export default ImageUploader
