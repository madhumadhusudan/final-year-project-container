import { ImageIcon } from './Icons.jsx'

function ImagePreview({ previewUrl, filename, onLoaded, onError }) {
  if (!previewUrl) {
    return <div className="grid min-h-72 place-items-center rounded-2xl bg-slate-100 text-slate-400" role="status"><div className="text-center"><ImageIcon className="mx-auto h-8 w-8" /><p className="mt-2 text-sm font-medium">Preparing image preview…</p></div></div>
  }

  return (
    <div className="flex min-h-72 max-h-[34rem] items-center justify-center overflow-hidden rounded-2xl bg-[linear-gradient(45deg,#f1f5f9_25%,transparent_25%),linear-gradient(-45deg,#f1f5f9_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#f1f5f9_75%),linear-gradient(-45deg,transparent_75%,#f1f5f9_75%)] bg-[length:20px_20px] bg-[position:0_0,0_10px,10px_-10px,-10px_0px] ring-1 ring-slate-200">
      <img src={previewUrl} alt={`Preview of ${filename}`} className="max-h-[34rem] w-full object-contain" onLoad={(event) => onLoaded({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} onError={onError} />
    </div>
  )
}

export default ImagePreview
