import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Footer from '../components/Footer.jsx'
import Navbar from '../components/Navbar.jsx'
import {
  cancelVideoJob, discardVideoUpload, getBackendHealth, getVideoJobStatus,
  getVideoResultUrl, startVideoProcessing, uploadVideo,
} from '../services/api.js'
import { formatBytes, formatDuration, getVideoFileError } from '../utils/videoFile.js'

const CATEGORY_CONTROLS = [
  ['protect_background_faces', 'background_faces', 'Background Faces'],
  ['protect_license_plates', 'license_plates', 'License Plates'],
  ['protect_cards', 'payment_cards', 'Payment Cards'],
  ['protect_identity_documents', 'identity_documents', 'Identity Documents'],
  ['protect_qr_codes', 'qr_codes', 'QR Codes'],
  ['protect_barcodes', 'barcodes', 'Barcodes'],
  ['protect_sensitive_text', 'sensitive_text', 'Sensitive Text'],
]

const DEFAULT_SETTINGS = {
  preserve_main_subject: true, protect_background_faces: true, protect_license_plates: true,
  protect_cards: true, protect_identity_documents: true, protect_qr_codes: true,
  protect_barcodes: true, protect_sensitive_text: true, anonymization_method: 'blur',
  strength: 'medium', quality_profile: 'balanced',
}

const PROFILE_TEXT = {
  performance: 'Fastest · faces every 5 frames · OCR every 30',
  balanced: 'Default · faces every 3 frames · OCR every 20',
  accuracy: 'More analysis · faces every 2 frames · OCR every 12',
}

function MetadataGrid({ metadata }) {
  const items = [
    ['Filename', metadata.filename], ['Duration', formatDuration(metadata.duration_seconds)],
    ['Resolution', `${metadata.width} × ${metadata.height}`], ['FPS', `${metadata.fps}${metadata.fps_fallback_used ? ' (safe fallback)' : ''}`],
    ['Frame count', metadata.frame_count ?? 'Unavailable'], ['File size', formatBytes(metadata.file_size_bytes)],
  ]
  return <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{items.map(([label, value]) => (
    <div key={label} className="rounded-xl border border-slate-200 bg-white p-3"><dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</dt><dd className="mt-1 truncate text-sm font-bold text-slate-900" title={String(value)}>{value}</dd></div>
  ))}</dl>
}

function VideoSummaryPanel({ summary }) {
  if (!summary) return null
  const risk = summary.risk
  return (
    <section className="mt-8 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6" aria-label="Video privacy report">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div><p className="text-sm font-semibold uppercase tracking-[0.16em] text-teal-700">Video Privacy Risk</p><h2 className="mt-1 text-2xl font-bold text-slate-950">Protection report</h2></div>
        <span className={`rounded-full px-3 py-1 text-sm font-bold ${summary.assessment === 'complete' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>{summary.assessment === 'complete' ? 'Complete assessment' : 'Partial privacy assessment'}</span>
      </div>
      {summary.unavailable_modules.length > 0 && <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">Unavailable modules: {summary.unavailable_modules.join(', ')}. The result is not labelled fully protected.</p>}
      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl bg-rose-50 p-4"><p className="text-xs font-bold uppercase text-rose-700">Original risk</p><p className="mt-1 text-3xl font-black text-rose-950">{risk.overall_video_score} <span className="text-sm">{risk.level}</span></p></div>
        <div className="rounded-xl bg-emerald-50 p-4"><p className="text-xs font-bold uppercase text-emerald-700">Residual risk</p><p className="mt-1 text-3xl font-black text-emerald-950">{risk.residual_score} <span className="text-sm">{risk.residual_level}</span></p></div>
        <div className="rounded-xl bg-teal-50 p-4"><p className="text-xs font-bold uppercase text-teal-700">Risk reduction</p><p className="mt-1 text-3xl font-black text-teal-950">{risk.risk_reduction}</p></div>
      </div>
      <p className="mt-3 text-xs text-slate-500">Category frequency is based on sampled detector frames, not every source frame.</p>
      <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm"><thead><tr className="border-b border-slate-200 text-slate-500"><th className="py-2">Category</th><th>Analyzed frames</th><th>Frames present</th><th>Presence</th></tr></thead><tbody>{Object.entries(risk.category_summary).map(([name, details]) => <tr key={name} className="border-b border-slate-100"><th className="py-2.5 font-semibold text-slate-800">{name}</th><td>{details.analyzed_frames}</td><td>{details.frames_present}</td><td>{details.presence_percent}%</td></tr>)}</tbody></table></div>
      <dl className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ['Processed frames', summary.total_processed_frames], ['Analyzed frames', summary.analyzed_frames],
          ['Protected regions', summary.protected_regions], ['Processing FPS', summary.performance.processing_fps],
          ['Processing time', `${summary.processing_time_seconds}s`], ['Output size', formatBytes(summary.output_size_bytes)],
          ['Codec', summary.codec], ['Resolution / FPS', `${summary.resolution} / ${summary.fps}`],
        ].map(([label, value]) => <div key={label} className="rounded-xl bg-slate-50 p-3"><dt className="text-xs font-semibold text-slate-500">{label}</dt><dd className="mt-1 font-bold text-slate-950">{value}</dd></div>)}
      </dl>
      <p className={`mt-4 rounded-xl p-3 text-sm ${summary.audio_preserved ? 'bg-emerald-50 text-emerald-900' : 'bg-slate-100 text-slate-700'}`}>{summary.audio_message}</p>
    </section>
  )
}

function VideoPrivacyPage() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [uploadState, setUploadState] = useState('idle')
  const [uploadError, setUploadError] = useState('')
  const [uploadData, setUploadData] = useState(null)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [job, setJob] = useState(null)
  const [jobError, setJobError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const uploadController = useRef(null)
  const pollController = useRef(null)
  const pollTimer = useRef(0)
  const jobRef = useRef(null)
  const uploadRef = useRef(null)

  useEffect(() => { jobRef.current = job }, [job])
  useEffect(() => { uploadRef.current = uploadData }, [uploadData])
  useEffect(() => {
    const controller = new AbortController()
    getBackendHealth(controller.signal).then(() => setBackendStatus('connected')).catch((error) => {
      if (error.name !== 'AbortError') setBackendStatus('offline')
    })
    return () => controller.abort()
  }, [])
  useEffect(() => {
    if (!file) return undefined
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => () => {
    uploadController.current?.abort()
    pollController.current?.abort()
    window.clearTimeout(pollTimer.current)
    const currentJob = jobRef.current
    if (currentJob && ['queued', 'processing'].includes(currentJob.state)) cancelVideoJob(currentJob.job_id).catch(() => {})
    if (uploadRef.current?.upload_id && !currentJob) discardVideoUpload(uploadRef.current.upload_id)
  }, [])

  const resetCurrent = useCallback(async () => {
    uploadController.current?.abort()
    pollController.current?.abort()
    window.clearTimeout(pollTimer.current)
    if (jobRef.current && ['queued', 'processing'].includes(jobRef.current.state)) await cancelVideoJob(jobRef.current.job_id).catch(() => {})
    if (uploadRef.current?.upload_id && !jobRef.current) await discardVideoUpload(uploadRef.current.upload_id)
    setJob(null); setUploadData(null); setUploadError(''); setJobError(''); setUploadState('idle')
  }, [])

  const chooseFile = useCallback(async (nextFile) => {
    const error = getVideoFileError(nextFile, uploadData?.capabilities?.max_size_bytes)
    if (error) { setUploadError(error); return }
    await resetCurrent()
    setFile(nextFile)
    setUploadState('uploading')
    const controller = new AbortController()
    uploadController.current = controller
    try {
      const response = await uploadVideo(nextFile, controller.signal)
      if (uploadController.current !== controller) return
      setUploadData(response)
      setUploadState('ready')
    } catch (requestError) {
      if (requestError.name !== 'AbortError' && uploadController.current === controller) {
        setUploadState('error'); setUploadError(requestError.message)
      }
    } finally {
      if (uploadController.current === controller) uploadController.current = null
    }
  }, [resetCurrent, uploadData?.capabilities?.max_size_bytes])

  const removeFile = useCallback(async () => {
    await resetCurrent()
    setFile(null); setPreviewUrl('')
  }, [resetCurrent])

  const poll = useCallback(async (jobId) => {
    const controller = new AbortController()
    pollController.current = controller
    try {
      const status = await getVideoJobStatus(jobId, controller.signal)
      setJob(status)
      if (['queued', 'processing'].includes(status.state)) pollTimer.current = window.setTimeout(() => poll(jobId), 700)
      else if (status.state === 'failed') setJobError(status.error || 'Video processing failed.')
    } catch (error) {
      if (error.name !== 'AbortError') setJobError(error.message)
    }
  }, [])

  const start = useCallback(async () => {
    if (!uploadData?.upload_id) return
    setJobError('')
    try {
      const response = await startVideoProcessing(uploadData.upload_id, settings)
      const initial = { ...response, progress: 0, stage: 'Preparing', output_available: false }
      setJob(initial)
      poll(response.job_id)
    } catch (error) {
      setJobError(error.message)
    }
  }, [uploadData, settings, poll])

  const cancel = useCallback(async () => {
    if (!job?.job_id) return
    window.clearTimeout(pollTimer.current)
    pollController.current?.abort()
    try { setJob(await cancelVideoJob(job.job_id)) } catch (error) { setJobError(error.message) }
  }, [job])

  const capabilities = uploadData?.capabilities
  const processing = ['queued', 'processing'].includes(job?.state)
  const resultUrl = job?.state === 'completed' ? getVideoResultUrl(job.job_id) : ''
  const profileDetails = useMemo(() => capabilities?.quality_profiles?.[settings.quality_profile], [capabilities, settings.quality_profile])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <Navbar backendStatus={backendStatus} videoMode />
      <main className="px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700">Local video workspace</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">Video Privacy Protection</h1>
          <p className="mt-3 max-w-2xl text-lg text-slate-600">Detect and anonymize privacy-sensitive content across video.</p>

          <div className="mt-8 grid items-start gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,.8fr)]">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-bold">1. Upload and validate</h2>
              <label onDragEnter={(e) => { e.preventDefault(); setDragActive(true) }} onDragOver={(e) => e.preventDefault()} onDragLeave={() => setDragActive(false)} onDrop={(e) => { e.preventDefault(); setDragActive(false); if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]) }} className={`mt-4 grid min-h-40 place-items-center rounded-2xl border-2 border-dashed p-6 text-center transition ${dragActive ? 'border-teal-600 bg-teal-50' : 'border-slate-300 bg-slate-50'}`}>
                <div><p className="font-bold text-slate-900">Drop a video here or browse</p><p className="mt-1 text-sm text-slate-500">MP4, MOV, AVI or WEBM · up to {Math.floor((capabilities?.max_size_bytes || 157286400) / 1024 / 1024)} MB</p><input className="mt-4 block w-full text-sm" type="file" accept=".mp4,.mov,.avi,.webm,video/mp4,video/quicktime,video/x-msvideo,video/webm" disabled={processing} onChange={(e) => e.target.files[0] && chooseFile(e.target.files[0])} /></div>
              </label>
              {uploadState === 'uploading' && <p className="mt-3 text-sm font-semibold text-teal-700">Uploading securely and reading real metadata…</p>}
              {uploadError && <p role="alert" className="mt-3 rounded-xl bg-rose-50 p-3 text-sm text-rose-800">{uploadError}</p>}
              {file && <div className="mt-4 flex justify-end"><button type="button" disabled={processing} onClick={removeFile} className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold disabled:opacity-50">Remove / change video</button></div>}
              {previewUrl && <video className="mt-4 aspect-video w-full rounded-xl bg-slate-950 object-contain" src={previewUrl} controls preload="metadata" />}
              {uploadData?.metadata && <div className="mt-5"><MetadataGrid metadata={uploadData.metadata} /></div>}
            </section>

            <aside className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-lg font-bold">2. Privacy settings</h2>
              <label className="mt-4 flex items-start gap-3 rounded-xl bg-teal-50 p-3"><input type="checkbox" className="mt-1 h-4 w-4" checked={settings.preserve_main_subject} disabled={processing} onChange={(e) => setSettings({ ...settings, preserve_main_subject: e.target.checked })} /><span><span className="block text-sm font-bold text-teal-950">Preserve Main Subject</span><span className="text-xs text-teal-800">Maintains one temporally stable face; uncertainty fails safe.</span></span></label>
              <fieldset className="mt-5"><legend className="text-sm font-bold">Protect</legend><div className="mt-2 space-y-2">{CATEGORY_CONTROLS.map(([setting, capability, label]) => {
                const available = capabilities?.modules?.[capability] !== false
                return <label key={setting} className={`flex items-center justify-between rounded-lg border p-2.5 text-sm ${available ? 'border-slate-200' : 'border-slate-100 bg-slate-50 text-slate-400'}`}><span>{label} {!available && <span className="text-xs">(unavailable)</span>}</span><input type="checkbox" checked={settings[setting] && available} disabled={processing || !available} onChange={(e) => setSettings({ ...settings, [setting]: e.target.checked })} /></label>
              })}</div></fieldset>
              <label className="mt-5 block text-sm font-bold">Method<select className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2.5 font-normal" disabled={processing} value={settings.anonymization_method} onChange={(e) => setSettings({ ...settings, anonymization_method: e.target.value })}><option value="blur">Blur</option><option value="pixelate">Pixelate</option><option value="blackout">Blackout</option></select></label>
              <label className="mt-4 block text-sm font-bold">Protection strength<select className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2.5 font-normal" disabled={processing} value={settings.strength} onChange={(e) => setSettings({ ...settings, strength: e.target.value })}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>
              <label className="mt-4 block text-sm font-bold">Quality profile<select className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2.5 font-normal" disabled={processing} value={settings.quality_profile} onChange={(e) => setSettings({ ...settings, quality_profile: e.target.value })}><option value="performance">Performance</option><option value="balanced">Balanced</option><option value="accuracy">Accuracy</option></select><span className="mt-1 block text-xs font-normal text-slate-500">{PROFILE_TEXT[settings.quality_profile]}{profileDetails ? ` · ${profileDetails.analysis_width}px analysis width` : ''}</span></label>
              <button type="button" onClick={start} disabled={uploadState !== 'ready' || processing || job?.state === 'completed'} className="mt-6 w-full rounded-xl bg-teal-700 px-4 py-3 font-bold text-white shadow-sm hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50">Start privacy analysis</button>
              {processing && <button type="button" onClick={cancel} className="mt-3 w-full rounded-xl border border-rose-300 px-4 py-2.5 font-bold text-rose-700">Cancel processing</button>}
            </aside>
          </div>

          {job && <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-live="polite"><div className="flex items-center justify-between gap-4"><div><p className="text-sm font-bold text-slate-900">{job.stage}</p><p className="text-xs capitalize text-slate-500">Job {job.state}</p></div><span className="font-black text-teal-800">{job.progress == null ? 'Working…' : `${Math.round(job.progress)}%`}</span></div>{job.progress == null ? <div className="mt-3 h-2 animate-pulse rounded-full bg-teal-200" /> : <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200"><div className="h-full rounded-full bg-teal-600 transition-[width]" style={{ width: `${job.progress}%` }} /></div>}</section>}
          {jobError && <p role="alert" className="mt-4 rounded-xl bg-rose-50 p-4 text-sm text-rose-800">{jobError}</p>}

          {resultUrl && <section className="mt-8 rounded-2xl border border-teal-200 bg-teal-950 p-5 text-white shadow-sm sm:p-6"><h2 className="text-xl font-bold">Protected video</h2><p className="mt-1 text-sm text-teal-100/80">Preview before downloading. Playback does not start automatically.</p><video className="mt-4 aspect-video w-full rounded-xl bg-black object-contain" src={resultUrl} controls preload="metadata" /><a href={resultUrl} download="privacy-protected-video.mp4" className="mt-4 inline-flex rounded-xl bg-white px-5 py-3 text-sm font-bold text-teal-950">Download Protected Video</a></section>}
          <VideoSummaryPanel summary={job?.summary} />
        </div>
      </main>
      <Footer />
    </div>
  )
}

export default VideoPrivacyPage
