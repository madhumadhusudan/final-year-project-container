import { formatFileSize, formatFileType } from '../utils/imageFile.js'

function ImageInfo({ metadata }) {
  const details = [
    { label: 'Filename', value: metadata.name },
    { label: 'Resolution', value: metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : 'Reading…' },
    { label: 'Size', value: formatFileSize(metadata.size) },
    { label: 'Type', value: formatFileType(metadata.type) },
  ]
  return (
    <dl className="mt-5 grid gap-3 sm:grid-cols-2">
      {details.map((detail) => <div key={detail.label} className="min-w-0 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"><dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">{detail.label}</dt><dd className="mt-1 truncate text-sm font-semibold text-slate-800" title={detail.value}>{detail.value}</dd></div>)}
    </dl>
  )
}

export default ImageInfo
