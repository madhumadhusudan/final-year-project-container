import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Footer from '../components/Footer.jsx'
import Navbar from '../components/Navbar.jsx'
import { analyzeFrame, getLiveCapabilities } from '../services/api.js'
import {
  RegionTracker, analysisToVideoCoordinates, calculateLiveRisk, modulesForRequest,
  paddedBox, videoToCanvasCoordinates, cameraFailureState, isFreshFrameResponse,
  shouldProtectTrack, stopMediaStream,
} from '../utils/livePrivacy.js'

const QUALITY_PROFILES = {
  performance: { width: 480, faces: 300, plates: 2200, cards: 3000, documents: 3500, codes: 1800, ocr: 4000 },
  balanced: { width: 640, faces: 180, plates: 1200, cards: 1800, documents: 2000, codes: 1000, ocr: 1800 },
  accuracy: { width: 768, faces: 130, plates: 900, cards: 1400, documents: 1600, codes: 700, ocr: 1400 },
}

const DEFAULT_CAPABILITIES = {
  background_faces: true, license_plates: false, payment_cards: false,
  identity_documents: false, qr_codes: false, barcodes: false, sensitive_text: false,
}

const CATEGORY_CONTROLS = [
  ['background_faces', 'Background Faces'], ['license_plates', 'License Plates'],
  ['payment_cards', 'Payment Cards'], ['identity_documents', 'Identity Documents'],
  ['qr_codes', 'QR Codes'], ['barcodes', 'Barcodes'], ['sensitive_text', 'Sensitive Text'],
]

const CATEGORY_SETTING = {
  face: 'background_faces', license_plate: 'license_plates', payment_card: 'payment_cards',
  identity_document: 'identity_documents', qr_code: 'qr_codes', barcode: 'barcodes',
  sensitive_text: 'sensitive_text',
}

const CATEGORY_LABEL = {
  face: 'background face', license_plate: 'license plate', payment_card: 'payment card',
  identity_document: 'identity document', qr_code: 'QR code', barcode: 'barcode',
  sensitive_text: 'sensitive text region',
}

function captureFrame(video, canvas, targetWidth) {
  const width = Math.min(targetWidth, video.videoWidth)
  const height = Math.max(1, Math.round(width * video.videoHeight / video.videoWidth))
  canvas.width = width
  canvas.height = height
  canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, width, height)
  return new Promise((resolve, reject) => canvas.toBlob(
    (blob) => (blob ? resolve(blob) : reject(new Error('Camera frame capture failed.'))),
    'image/jpeg', 0.72,
  ))
}

function protectRegion(context, video, box, method, strength, pixelCanvas) {
  const x = Math.floor(box.x1)
  const y = Math.floor(box.y1)
  const width = Math.max(1, Math.ceil(box.x2 - box.x1))
  const height = Math.max(1, Math.ceil(box.y2 - box.y1))
  if (method === 'blackout') {
    context.save()
    context.fillStyle = '#020617'
    context.fillRect(x, y, width, height)
    context.restore()
    return
  }
  if (method === 'pixelate') {
    const divisor = { low: 9, medium: 14, high: 20 }[strength]
    const smallWidth = Math.max(1, Math.round(width / divisor))
    const smallHeight = Math.max(1, Math.round(height / divisor))
    pixelCanvas.width = smallWidth
    pixelCanvas.height = smallHeight
    const pixelContext = pixelCanvas.getContext('2d', { alpha: false })
    pixelContext.imageSmoothingEnabled = true
    pixelContext.drawImage(video, x, y, width, height, 0, 0, smallWidth, smallHeight)
    context.save()
    context.imageSmoothingEnabled = false
    context.drawImage(pixelCanvas, 0, 0, smallWidth, smallHeight, x, y, width, height)
    context.restore()
    return
  }
  context.save()
  context.beginPath()
  context.rect(x, y, width, height)
  context.clip()
  context.filter = `blur(${{ low: 9, medium: 15, high: 23 }[strength]}px)`
  context.drawImage(video, 0, 0, context.canvas.width, context.canvas.height)
  context.restore()
}

function Metric({ label, value }) {
  return <div className="rounded-xl bg-slate-50 p-3"><dt className="text-xs font-semibold text-slate-500">{label}</dt><dd className="mt-1 text-lg font-bold text-slate-950">{value}</dd></div>
}

function LivePrivacyPage() {
  const [cameraState, setCameraState] = useState('idle')
  const [cameraError, setCameraError] = useState('')
  const [detectionState, setDetectionState] = useState('idle')
  const [capabilities, setCapabilities] = useState(DEFAULT_CAPABILITIES)
  const [capabilitiesLoaded, setCapabilitiesLoaded] = useState(false)
  const [facingMode, setFacingMode] = useState('user')
  const [cameraCount, setCameraCount] = useState(0)
  const [preserveMainSubject, setPreserveMainSubject] = useState(true)
  const [manualMainTrackId, setManualMainTrackId] = useState(null)
  const [mainTrackId, setMainTrackId] = useState(null)
  const [method, setMethod] = useState('blur')
  const [strength, setStrength] = useState('medium')
  const [quality, setQuality] = useState('balanced')
  const [enabled, setEnabled] = useState(() => Object.fromEntries(CATEGORY_CONTROLS.map(([key]) => [key, true])))
  const [tracks, setTracks] = useState([])
  const [metrics, setMetrics] = useState({ renderFps: 0, analysisFps: 0, latestLatency: 0, averageLatency: 0 })

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const captureCanvasRef = useRef(document.createElement('canvas'))
  const pixelCanvasRef = useRef(document.createElement('canvas'))
  const streamRef = useRef(null)
  const renderFrameRef = useRef(0)
  const analysisTimerRef = useRef(0)
  const analysisControllerRef = useRef(null)
  const inFlightRef = useRef(false)
  const runTokenRef = useRef(0)
  const frameIdRef = useRef(0)
  const lastAcceptedFrameRef = useRef(0)
  const lastRunRef = useRef({})
  const trackerRef = useRef(new RegionTracker())
  const cameraStateRef = useRef(cameraState)
  const detectionStateRef = useRef(detectionState)
  const capabilitiesRef = useRef(capabilities)
  const settingsRef = useRef({ enabled, method, strength, preserveMainSubject, mainTrackId })
  const manualMainTrackIdRef = useRef(manualMainTrackId)
  const profileRef = useRef(QUALITY_PROFILES[quality])
  const effectiveFaceIntervalRef = useRef(QUALITY_PROFILES[quality].faces)
  const visibleRef = useRef(!document.hidden)
  const renderStatsRef = useRef({ started: performance.now(), frames: 0 })
  const analysisTimesRef = useRef([])
  const analysisStartedAtRef = useRef(performance.now())
  const latenciesRef = useRef([])

  useEffect(() => { cameraStateRef.current = cameraState }, [cameraState])
  useEffect(() => { detectionStateRef.current = detectionState }, [detectionState])
  useEffect(() => { capabilitiesRef.current = capabilities }, [capabilities])
  useEffect(() => { settingsRef.current = { enabled, method, strength, preserveMainSubject, mainTrackId } }, [enabled, method, strength, preserveMainSubject, mainTrackId])
  useEffect(() => { manualMainTrackIdRef.current = manualMainTrackId }, [manualMainTrackId])
  useEffect(() => {
    profileRef.current = QUALITY_PROFILES[quality]
    effectiveFaceIntervalRef.current = QUALITY_PROFILES[quality].faces
  }, [quality])

  useEffect(() => {
    const controller = new AbortController()
    getLiveCapabilities(controller.signal).then((data) => {
      setCapabilities({ ...DEFAULT_CAPABILITIES, ...data.modules })
      setCapabilitiesLoaded(true)
    }).catch((error) => {
      if (error.name !== 'AbortError') {
        setCapabilitiesLoaded(true)
        setDetectionState('unavailable')
      }
    })
    return () => controller.abort()
  }, [])

  const releaseResources = useCallback((updateState = true) => {
    runTokenRef.current += 1
    window.clearTimeout(analysisTimerRef.current)
    window.cancelAnimationFrame(renderFrameRef.current)
    analysisControllerRef.current?.abort()
    analysisControllerRef.current = null
    inFlightRef.current = false
    stopMediaStream(streamRef.current)
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }
    for (const canvas of [canvasRef.current, captureCanvasRef.current, pixelCanvasRef.current]) {
      if (!canvas) continue
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      if (canvas !== canvasRef.current) canvas.width = canvas.height = 1
    }
    trackerRef.current.clear()
    setTracks([])
    setMainTrackId(null)
    setManualMainTrackId(null)
    setMetrics({ renderFps: 0, analysisFps: 0, latestLatency: 0, averageLatency: 0 })
    lastAcceptedFrameRef.current = 0
    if (updateState) {
      cameraStateRef.current = 'idle'
      setCameraState('idle')
      setDetectionState('idle')
    }
  }, [])

  const renderLoop = useCallback((token) => {
    if (token !== runTokenRef.current || cameraStateRef.current !== 'active') return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (video?.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
      const context = canvas.getContext('2d', { alpha: false })
      context.filter = 'none'
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const settings = settingsRef.current
      if (detectionStateRef.current !== 'active') {
        context.save()
        context.filter = 'blur(24px)'
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        context.fillStyle = 'rgba(2, 6, 23, 0.22)'
        context.fillRect(0, 0, canvas.width, canvas.height)
        context.restore()
      } else {
        const currentTracks = trackerRef.current.active(performance.now())
        for (const track of currentTracks) {
          if (!shouldProtectTrack(track, settings, settings.mainTrackId)) continue
          const canvasBox = videoToCanvasCoordinates(
            track.box,
            { width: video.videoWidth, height: video.videoHeight },
            { width: canvas.width, height: canvas.height },
          )
          protectRegion(context, video, paddedBox(canvasBox, track.category, canvas), settings.method, settings.strength, pixelCanvasRef.current)
        }
      }
      const stats = renderStatsRef.current
      stats.frames += 1
      const elapsed = performance.now() - stats.started
      if (elapsed >= 750) {
        const renderFps = stats.frames * 1000 / elapsed
        setMetrics((current) => ({ ...current, renderFps }))
        renderStatsRef.current = { started: performance.now(), frames: 0 }
      }
    }
    renderFrameRef.current = window.requestAnimationFrame(() => renderLoop(token))
  }, [])

  const analysisLoop = useCallback(async (token) => {
    if (token !== runTokenRef.current || cameraStateRef.current !== 'active') return
    const schedule = (delay = effectiveFaceIntervalRef.current) => {
      analysisTimerRef.current = window.setTimeout(() => analysisLoop(token), Math.max(20, delay))
    }
    if (!visibleRef.current || inFlightRef.current) {
      schedule(250)
      return
    }
    const video = videoRef.current
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) {
      schedule(100)
      return
    }
    inFlightRef.current = true
    const requestStarted = performance.now()
    const profile = profileRef.current
    const now = performance.now()
    const requestModules = modulesForRequest(now, lastRunRef.current, profile, capabilitiesRef.current)
    requestModules.forEach((name) => { lastRunRef.current[name] = now })
    const frameId = ++frameIdRef.current
    const capturedAtMs = Date.now()
    const controller = new AbortController()
    analysisControllerRef.current = controller
    const timeout = window.setTimeout(() => controller.abort('timeout'), 8000)
    try {
      const blob = await captureFrame(video, captureCanvasRef.current, profile.width)
      if (token !== runTokenRef.current) return
      const response = await analyzeFrame(blob, {
        frameId, capturedAtMs, modules: requestModules,
        preserveMainSubject: settingsRef.current.preserveMainSubject,
      }, controller.signal)
      if (token !== runTokenRef.current || !isFreshFrameResponse(response.frame_id, lastAcceptedFrameRef.current)) return
      lastAcceptedFrameRef.current = response.frame_id
      const completedModules = requestModules.filter((name) => response.modules?.[name]?.status === 'completed')
      const mappedRegions = response.regions.map((region) => ({
        ...region,
        bounding_box: analysisToVideoCoordinates(
          region.bounding_box, response.image,
          { width: video.videoWidth, height: video.videoHeight },
        ),
      }))
      const currentTracks = trackerRef.current.update(mappedRegions, completedModules, performance.now())
      const faceStatus = response.modules?.faces?.status
      if (faceStatus === 'completed') {
        detectionStateRef.current = 'active'
        setDetectionState('active')
      } else if (faceStatus === 'error' || faceStatus === 'unavailable') {
        detectionStateRef.current = 'unavailable'
        setDetectionState('unavailable')
      }
      let selectedMain = settingsRef.current.mainTrackId
      const manualId = manualMainTrackIdRef.current
      const manual = manualId && currentTracks.some((track) => track.trackId === manualId)
      if (manual) selectedMain = manualId
      else {
        if (manualId) setManualMainTrackId(null)
        if (!currentTracks.some((track) => track.trackId === selectedMain && track.category === 'face')) {
          selectedMain = currentTracks.find((track) => track.category === 'face' && track.role === 'main_subject')?.trackId || null
        }
      }
      setMainTrackId(selectedMain)
      setTracks([...currentTracks])

      const latency = performance.now() - requestStarted
      const completion = performance.now()
      analysisTimesRef.current = [...analysisTimesRef.current.filter((value) => completion - value <= 5000), completion]
      latenciesRef.current = [...latenciesRef.current.slice(-19), latency]
      const averageLatency = latenciesRef.current.reduce((sum, value) => sum + value, 0) / latenciesRef.current.length
      setMetrics((current) => ({
        ...current, latestLatency: latency, averageLatency,
        analysisFps: analysisTimesRef.current.length * 1000 / Math.max(1000, Math.min(5000, completion - analysisStartedAtRef.current)),
      }))
      const baseInterval = profile.faces
      if (latency > effectiveFaceIntervalRef.current * 1.8) {
        effectiveFaceIntervalRef.current = Math.min(600, effectiveFaceIntervalRef.current * 1.25)
      } else if (latency < effectiveFaceIntervalRef.current * 0.7) {
        effectiveFaceIntervalRef.current = Math.max(baseInterval, effectiveFaceIntervalRef.current * 0.95)
      }
    } catch (error) {
      if (token === runTokenRef.current && error.name !== 'AbortError') {
        detectionStateRef.current = 'unavailable'
        setDetectionState('unavailable')
      } else if (token === runTokenRef.current && controller.signal.reason === 'timeout') {
        detectionStateRef.current = 'unavailable'
        setDetectionState('unavailable')
      }
    } finally {
      window.clearTimeout(timeout)
      if (analysisControllerRef.current === controller) analysisControllerRef.current = null
      inFlightRef.current = false
      if (token === runTokenRef.current) schedule(effectiveFaceIntervalRef.current - (performance.now() - requestStarted))
    }
  }, [])

  const startCamera = useCallback(async (nextFacingMode = facingMode) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('camera_unavailable')
      setCameraError('This browser does not provide camera access. Use a modern browser in a secure context.')
      return
    }
    releaseResources(false)
    const token = runTokenRef.current
    setCameraError('')
    cameraStateRef.current = 'requesting_permission'
    setCameraState('requesting_permission')
    setDetectionState('idle')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: nextFacingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      if (token !== runTokenRef.current) {
        stopMediaStream(stream)
        return
      }
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      const devices = await navigator.mediaDevices.enumerateDevices().catch(() => [])
      setCameraCount(devices.filter((device) => device.kind === 'videoinput').length)
      setFacingMode(nextFacingMode)
      trackerRef.current.clear()
      const profile = profileRef.current
      const started = performance.now()
      lastRunRef.current = { plates: started, cards: started, documents: started, codes: started, ocr: started }
      renderStatsRef.current = { started, frames: 0 }
      analysisTimesRef.current = []
      analysisStartedAtRef.current = started
      latenciesRef.current = []
      cameraStateRef.current = 'active'
      setCameraState('active')
      renderLoop(token)
      analysisTimerRef.current = window.setTimeout(() => analysisLoop(token), Math.min(80, profile.faces))
    } catch (error) {
      if (token !== runTokenRef.current) return
      releaseResources(false)
      const failure = cameraFailureState(error.name)
      setCameraState(failure.state)
      setCameraError(failure.message)
    }
  }, [analysisLoop, facingMode, releaseResources, renderLoop])

  const stopCamera = useCallback(() => {
    if (cameraStateRef.current === 'idle') return
    cameraStateRef.current = 'stopping'
    setCameraState('stopping')
    releaseResources(true)
  }, [releaseResources])

  useEffect(() => {
    const onVisibility = () => {
      visibleRef.current = !document.hidden
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      releaseResources(false)
    }
  }, [releaseResources])

  const faceTracks = tracks.filter((track) => track.category === 'face')
  const risk = useMemo(() => calculateLiveRisk(tracks, preserveMainSubject, mainTrackId), [tracks, preserveMainSubject, mainTrackId])
  const warnings = useMemo(() => {
    const counts = {}
    tracks.forEach((track) => {
      if (track.category === 'face' && preserveMainSubject && track.trackId === mainTrackId) return
      if (!enabled[CATEGORY_SETTING[track.category]]) return
      counts[track.category] = (counts[track.category] || 0) + 1
    })
    return Object.entries(counts).map(([category, count]) => `${count} ${CATEGORY_LABEL[category]}${count === 1 || category === 'sensitive_text' ? '' : 's'} protected`)
  }, [enabled, mainTrackId, preserveMainSubject, tracks])
  const isActive = cameraState === 'active'
  const protectionActive = isActive && detectionState === 'active'

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <Navbar backendStatus={detectionState === 'active' ? 'connected' : detectionState === 'unavailable' ? 'offline' : 'checking'} liveMode />
      <main className="px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
            <div><p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700">On-device camera mode</p><h1 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">Live Privacy Protection</h1><p className="mt-3 max-w-2xl leading-7 text-slate-600">Detect and protect sensitive information before it appears on screen.</p></div>
            <div className="flex flex-wrap gap-2">
              {!isActive ? <button type="button" disabled={cameraState === 'requesting_permission' || cameraState === 'stopping'} onClick={() => startCamera()} className="rounded-xl bg-teal-700 px-5 py-3 text-sm font-bold text-white shadow-sm hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-60">{cameraState === 'requesting_permission' ? 'Requesting permission…' : 'Start Camera'}</button>
                : <button type="button" onClick={stopCamera} className="rounded-xl bg-rose-700 px-5 py-3 text-sm font-bold text-white hover:bg-rose-800">Stop Camera</button>}
              {isActive && cameraCount > 1 && <button type="button" onClick={() => startCamera(facingMode === 'user' ? 'environment' : 'user')} className="rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-bold text-slate-700 hover:bg-slate-100">Use {facingMode === 'user' ? 'Back' : 'Front'} Camera</button>}
            </div>
          </div>

          {cameraError && <div role="alert" className="mt-6 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm font-semibold text-rose-900">{cameraError}</div>}
          <div className="mt-7 grid items-start gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(330px,.75fr)]">
            <section className="overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 shadow-sm" aria-label="Protected live preview">
              <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 text-white">
                <div className="flex items-center gap-3"><span className={`h-2.5 w-2.5 rounded-full ${isActive ? 'animate-pulse bg-rose-500' : 'bg-slate-500'}`} /><span className="text-sm font-bold">{isActive ? 'LIVE · Protected preview' : 'Camera stopped'}</span></div>
                <span className="text-xs text-slate-300">Raw feed hidden</span>
              </div>
              <div className="relative grid min-h-[280px] place-items-center bg-slate-900 sm:min-h-[420px]">
                <video ref={videoRef} muted playsInline className="pointer-events-none absolute h-px w-px opacity-0" aria-hidden="true" />
                <canvas ref={canvasRef} className={`block max-h-[72vh] w-full object-contain ${isActive ? '' : 'hidden'}`} onClick={(event) => {
                  if (!preserveMainSubject) return
                  const canvas = event.currentTarget
                  const rect = canvas.getBoundingClientRect()
                  const x = (event.clientX - rect.left) * canvas.width / rect.width
                  const y = (event.clientY - rect.top) * canvas.height / rect.height
                  const selected = faceTracks.find((track) => x >= track.box.x1 && x <= track.box.x2 && y >= track.box.y1 && y <= track.box.y2)
                  if (selected) { setManualMainTrackId(selected.trackId); setMainTrackId(selected.trackId) }
                }} aria-label="Protected camera output. Click a detected face to preserve it." />
                {!isActive && <div className="px-6 text-center text-slate-300"><div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-white/10 bg-white/5 text-2xl">◉</div><p className="mt-4 font-bold text-white">Camera access starts only when you ask</p><p className="mt-2 max-w-md text-sm leading-6 text-slate-400">Video only. No microphone, recording, cloud upload, or permanent frame storage.</p></div>}
                {isActive && detectionState !== 'active' && <div className="pointer-events-none absolute inset-x-4 bottom-4 rounded-xl bg-amber-950/90 px-4 py-3 text-center text-sm font-bold text-amber-100">Safety blur active while live detection is {detectionState === 'unavailable' ? 'unavailable' : 'starting'}.</div>}
              </div>
            </section>

            <aside className="space-y-5">
              <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="font-bold">Live status</h2><dl className="mt-4 grid grid-cols-3 gap-2 text-center"><Metric label="Camera" value={isActive ? 'Active' : 'Stopped'} /><Metric label="Detection" value={detectionState === 'active' ? 'Active' : detectionState === 'unavailable' ? 'Unavailable' : 'Waiting'} /><Metric label="Protection" value={protectionActive ? method[0].toUpperCase() + method.slice(1) : detectionState === 'unavailable' && isActive ? 'Safety blur' : 'Off'} /></dl>{isActive && detectionState === 'unavailable' && <p className="mt-3 rounded-xl bg-rose-50 p-3 text-sm font-semibold text-rose-900">Live privacy detection unavailable. The full preview remains blurred.</p>}</section>

              <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><h2 className="font-bold">Privacy controls</h2><span className="text-xs font-semibold text-slate-500">Local only</span></div><label className="mt-4 flex items-start gap-3 rounded-xl bg-teal-50 p-3"><input type="checkbox" checked={preserveMainSubject} onChange={(event) => setPreserveMainSubject(event.target.checked)} className="mt-1 h-4 w-4 accent-teal-700" /><span><span className="block text-sm font-bold text-teal-950">Preserve Main Subject</span><span className="text-xs leading-5 text-teal-800">Keeps one spatially tracked face clear. This is not face recognition.</span></span></label>
                <fieldset className="mt-4 space-y-2"><legend className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">Information to protect</legend>{CATEGORY_CONTROLS.map(([key, label]) => { const available = capabilities[key]; return <label key={key} className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${available ? 'border-slate-200' : 'border-slate-100 bg-slate-50 text-slate-400'}`}><span><input type="checkbox" disabled={!available} checked={available && enabled[key]} onChange={(event) => setEnabled((current) => ({ ...current, [key]: event.target.checked }))} className="mr-2 accent-teal-700" />{label}</span>{!available && capabilitiesLoaded && <span className="text-[10px] font-bold uppercase">Unavailable</span>}</label>})}</fieldset>
                <fieldset className="mt-5"><legend className="text-xs font-bold uppercase tracking-wide text-slate-500">Protection method</legend><div className="mt-2 grid grid-cols-3 gap-2">{['blur', 'pixelate', 'blackout'].map((value) => <label key={value} className={`rounded-lg border p-2 text-center text-xs font-bold ${method === value ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-slate-200'}`}><input type="radio" name="live-method" value={value} checked={method === value} onChange={() => setMethod(value)} className="sr-only" />{value[0].toUpperCase() + value.slice(1)}</label>)}</div></fieldset>
                {method !== 'blackout' && <label className="mt-4 block text-xs font-bold uppercase tracking-wide text-slate-500">Strength<select value={strength} onChange={(event) => setStrength(event.target.value)} className="mt-2 block w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm normal-case text-slate-900"><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>}
                <label className="mt-4 block text-xs font-bold uppercase tracking-wide text-slate-500">Quality mode<select value={quality} onChange={(event) => setQuality(event.target.value)} className="mt-2 block w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm normal-case text-slate-900"><option value="performance">Performance · 480px</option><option value="balanced">Balanced · 640px</option><option value="accuracy">Accuracy · 768px</option></select></label>
              </section>
            </aside>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-3">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-wide text-slate-500">Live Privacy Risk</p><div className="mt-2 flex items-end gap-3"><span className="text-4xl font-black">{risk.score}<span className="text-lg text-slate-400"> / 100</span></span><span className="mb-1 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-black text-amber-900">{risk.level}</span></div><p className="mt-3 text-xs leading-5 text-slate-500">Updated from currently tracked detector evidence; mildly stabilized by region tracking.</p></section>
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="font-bold">Live warnings</h2>{warnings.length ? <ul className="mt-3 space-y-2">{warnings.map((warning) => <li key={warning} className="rounded-lg bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-950">{warning}</li>)}</ul> : <p className="mt-3 text-sm text-slate-500">No privacy regions are currently tracked.</p>}{preserveMainSubject && faceTracks.length > 1 && <div className="mt-4"><p className="text-xs font-bold uppercase text-slate-500">Set main subject</p><div className="mt-2 flex flex-wrap gap-2">{faceTracks.map((face, index) => <button type="button" key={face.trackId} onClick={() => { setManualMainTrackId(face.trackId); setMainTrackId(face.trackId) }} className={`rounded-lg px-3 py-2 text-xs font-bold ${mainTrackId === face.trackId ? 'bg-teal-700 text-white' : 'bg-slate-100 text-slate-700'}`}>Face {index + 1}</button>)}</div></div>}</section>
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><h2 className="font-bold">Performance</h2><span className="text-xs font-semibold text-slate-500">Adaptive</span></div><dl className="mt-4 grid grid-cols-2 gap-2"><Metric label="Render FPS" value={metrics.renderFps.toFixed(1)} /><Metric label="Analysis FPS" value={metrics.analysisFps.toFixed(1)} /><Metric label="Latest latency" value={`${metrics.latestLatency.toFixed(0)} ms`} /><Metric label="Average latency" value={`${metrics.averageLatency.toFixed(0)} ms`} /></dl><p className="mt-3 text-xs leading-5 text-slate-500">Face target: {Math.round(1000 / effectiveFaceIntervalRef.current)} checks/s · OCR: every {(QUALITY_PROFILES[quality].ocr / 1000).toFixed(1)}s when available.</p></section>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}

export default LivePrivacyPage
