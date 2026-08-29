import { useCallback, useEffect, useRef, useState } from 'react'
import AnalysisPanel from './components/AnalysisPanel.jsx'
import BeforeAfterPanel from './components/BeforeAfterPanel.jsx'
import Footer from './components/Footer.jsx'
import HeroSection from './components/HeroSection.jsx'
import HowItWorks from './components/HowItWorks.jsx'
import Navbar from './components/Navbar.jsx'
import PrivacyControls from './components/PrivacyControls.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'
import { analyzeImage, getBackendHealth, protectImage } from './services/api.js'
import { getImageFileError } from './utils/imageFile.js'

const defaultPrivacySettings = {
  protect_background_faces: true,
  protect_license_plates: true,
  protect_cards: true,
  protect_identity_documents: true,
  protect_sensitive_text: true,
  anonymization_method: 'blur',
  strength: 'medium',
}

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [imageMetadata, setImageMetadata] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [analysisStatus, setAnalysisStatus] = useState('idle')
  const [analysisResult, setAnalysisResult] = useState(null)
  const [analysisError, setAnalysisError] = useState('')
  const [privacySettings, setPrivacySettings] = useState(defaultPrivacySettings)
  const [protectionStatus, setProtectionStatus] = useState('idle')
  const [protectedBlob, setProtectedBlob] = useState(null)
  const [protectedUrl, setProtectedUrl] = useState('')
  const [protectionMetadata, setProtectionMetadata] = useState(null)
  const [protectionError, setProtectionError] = useState('')
  const analysisControllerRef = useRef(null)
  const protectionControllerRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    async function checkBackend() {
      try {
        await getBackendHealth(controller.signal)
        setBackendStatus('connected')
      } catch (error) {
        if (error.name !== 'AbortError') setBackendStatus('offline')
      }
    }
    checkBackend()
    return () => controller.abort()
  }, [])

  useEffect(() => () => {
    analysisControllerRef.current?.abort()
    protectionControllerRef.current?.abort()
  }, [])

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      return undefined
    }
    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  useEffect(() => {
    if (!protectedBlob) {
      setProtectedUrl('')
      return undefined
    }
    const objectUrl = URL.createObjectURL(protectedBlob)
    setProtectedUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [protectedBlob])

  const clearProtection = useCallback(() => {
    protectionControllerRef.current?.abort()
    setProtectionStatus('idle')
    setProtectedBlob(null)
    setProtectionMetadata(null)
    setProtectionError('')
  }, [])

  const selectImage = useCallback((file) => {
    const error = getImageFileError(file)
    if (error) {
      setUploadError(error)
      return
    }
    analysisControllerRef.current?.abort()
    clearProtection()
    setUploadError('')
    setAnalysisStatus('idle')
    setAnalysisResult(null)
    setAnalysisError('')
    setImageMetadata({ name: file.name, type: file.type, size: file.size, width: null, height: null })
    setSelectedFile(file)
  }, [clearProtection])

  const removeImage = useCallback(() => {
    analysisControllerRef.current?.abort()
    clearProtection()
    setSelectedFile(null)
    setImageMetadata(null)
    setUploadError('')
    setAnalysisStatus('idle')
    setAnalysisResult(null)
    setAnalysisError('')
  }, [clearProtection])

  const handleImageLoaded = useCallback(({ width, height }) => {
    setImageMetadata((current) => (current ? { ...current, width, height } : current))
  }, [])

  const handleImageError = useCallback(() => {
    clearProtection()
    setSelectedFile(null)
    setImageMetadata(null)
    setAnalysisStatus('idle')
    setAnalysisResult(null)
    setAnalysisError('')
    setUploadError('This image could not be previewed. Please choose a valid JPG, PNG or WEBP image.')
  }, [clearProtection])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return
    analysisControllerRef.current?.abort()
    clearProtection()
    const controller = new AbortController()
    analysisControllerRef.current = controller
    setAnalysisStatus('analyzing')
    setAnalysisResult(null)
    setAnalysisError('')
    try {
      const result = await analyzeImage(selectedFile, controller.signal)
      if (analysisControllerRef.current !== controller) return
      setAnalysisResult(result)
      setAnalysisStatus('success')
    } catch (error) {
      if (error.name === 'AbortError' || analysisControllerRef.current !== controller) return
      setAnalysisError(error.message || 'Image analysis failed. Please try again.')
      setAnalysisStatus('error')
    } finally {
      if (analysisControllerRef.current === controller) analysisControllerRef.current = null
    }
  }, [selectedFile, clearProtection])

  const handleProtect = useCallback(async () => {
    if (!selectedFile || !analysisResult || analysisStatus !== 'success') return
    protectionControllerRef.current?.abort()
    const controller = new AbortController()
    protectionControllerRef.current = controller
    setProtectionStatus('protecting')
    setProtectionError('')
    try {
      const result = await protectImage(selectedFile, analysisResult, privacySettings, controller.signal)
      if (protectionControllerRef.current !== controller) return
      setProtectedBlob(result.blob)
      setProtectionMetadata(result.protection)
      setProtectionStatus('success')
    } catch (error) {
      if (error.name === 'AbortError' || protectionControllerRef.current !== controller) return
      setProtectionError(error.message || 'Privacy protection failed. Please try again.')
      setProtectionStatus('error')
    } finally {
      if (protectionControllerRef.current === controller) protectionControllerRef.current = null
    }
  }, [selectedFile, analysisResult, analysisStatus, privacySettings])

  const handlePrivacySettingsChange = useCallback((nextSettings) => {
    clearProtection()
    setPrivacySettings(nextSettings)
  }, [clearProtection])

  const handleDownload = useCallback(() => {
    if (!protectedUrl || !selectedFile) return
    const stem = selectedFile.name.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '') || 'image'
    const extension = protectedBlob?.type === 'image/png' ? 'png' : protectedBlob?.type === 'image/webp' ? 'webp' : 'jpg'
    const link = document.createElement('a')
    link.href = protectedUrl
    link.download = `privacy-protected-${stem}.${extension}`
    document.body.appendChild(link)
    link.click()
    link.remove()
  }, [protectedUrl, protectedBlob, selectedFile])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <Navbar backendStatus={backendStatus} />
      <main>
        <HeroSection />
        <section id="analyze" className="scroll-mt-24 px-4 pb-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="mb-7 max-w-2xl">
              <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700">Privacy workspace</p>
              <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">Check an image before it goes public</h2>
              <p className="mt-2 leading-7 text-slate-600">Your image is sent only to your local detection service, processed transiently, and never retained.</p>
            </div>
            <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.75fr)]">
              <AnalysisPanel selectedFile={selectedFile} previewUrl={previewUrl} imageMetadata={imageMetadata} uploadError={uploadError} analysisStatus={analysisStatus} result={analysisResult} onFileSelect={selectImage} onRemove={removeImage} onImageLoaded={handleImageLoaded} onImageError={handleImageError} onAnalyze={handleAnalyze} />
              <PrivacyControls settings={privacySettings} onChange={handlePrivacySettingsChange} onProtect={handleProtect} canProtect={Boolean(selectedFile && analysisStatus === 'success')} protectionStatus={protectionStatus} error={protectionError} analysis={analysisResult?.analysis} />
            </div>
            <ResultsPanel analysisStatus={analysisStatus} hasImage={Boolean(selectedFile)} result={analysisResult} error={analysisError} />
            <BeforeAfterPanel originalUrl={previewUrl} protectedUrl={protectedUrl} filename={selectedFile?.name || 'image'} protection={protectionMetadata} onDownload={handleDownload} />
          </div>
        </section>
        <HowItWorks />
        <section id="privacy" className="scroll-mt-24 px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto flex max-w-7xl flex-col gap-6 rounded-3xl border border-teal-100 bg-teal-950 px-6 py-8 text-white shadow-sm sm:px-10 sm:py-10 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-2xl"><p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-300">Built around privacy</p><h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Your photo should be checked before it is shared.</h2><p className="mt-3 leading-7 text-teal-50/75">Detection and selective anonymization run on your local FastAPI service. Images are processed transiently and never forwarded to cloud AI APIs.</p></div>
            <a href="#analyze" className="inline-flex shrink-0 items-center justify-center rounded-xl bg-white px-5 py-3 text-sm font-semibold text-teal-950 shadow-sm transition hover:bg-teal-50 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white">Choose an image</a>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}

export default App
