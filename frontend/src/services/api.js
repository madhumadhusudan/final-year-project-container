const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export async function getBackendHealth(signal) {
  const response = await fetch(`${API_BASE_URL}/health`, {
    headers: { Accept: 'application/json' },
    signal,
  })

  if (!response.ok) {
    throw new Error(`Backend health check failed with status ${response.status}`)
  }

  const data = await response.json()
  if (data.status !== 'healthy') {
    throw new Error('Backend returned an unhealthy status')
  }

  return data
}

export async function getLiveCapabilities(signal) {
  const response = await fetch(`${API_BASE_URL}/live/capabilities`, {
    headers: { Accept: 'application/json' }, signal,
  })
  if (!response.ok) throw new Error('Live detector capabilities are unavailable.')
  return response.json()
}

export async function analyzeFrame(blob, request, signal) {
  const formData = new FormData()
  formData.append('image', blob, `frame-${request.frameId}.jpg`)
  formData.append('frame_id', String(request.frameId))
  formData.append('captured_at_ms', String(request.capturedAtMs))
  formData.append('modules', request.modules.join(','))
  formData.append('preserve_main_subject', String(request.preserveMainSubject))
  let response
  try {
    response = await fetch(`${API_BASE_URL}/analyze-frame`, {
      method: 'POST', body: formData, headers: { Accept: 'application/json' }, signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error('Live privacy detection unavailable.')
  }
  if (!response.ok) {
    const message = await getErrorMessage(response)
    throw new Error(message === 'Image analysis failed. Please check the image and try again.'
      ? 'Live privacy detection unavailable.' : message)
  }
  return response.json()
}

async function getErrorMessage(response) {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
  } catch {
    // Use a safe status-based message when the server did not return JSON.
  }
  if (response.status === 413) return 'The selected image is too large.'
  if (response.status === 503) return 'The local detection engine is unavailable. Please try again.'
  return 'Image analysis failed. Please check the image and try again.'
}

export async function analyzeImage(file, options, signal) {
  const formData = new FormData()
  formData.append('image', file, file.name)
  formData.append('analysis_options', JSON.stringify(options))
  let response
  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST', body: formData,
      headers: { Accept: 'application/json' }, signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error('The local backend is offline. Start it and try again.')
  }

  if (!response.ok) throw new Error(await getErrorMessage(response))
  return response.json()
}

export async function protectImage(file, analysis, settings, signal) {
  const formData = new FormData()
  formData.append('image', file, file.name)
  if (analysis.analysis_id) formData.append('analysis_id', analysis.analysis_id)
  else formData.append('analysis', JSON.stringify(analysis))
  formData.append('settings', JSON.stringify(settings))
  let response
  try {
    response = await fetch(`${API_BASE_URL}/protect`, {
      method: 'POST', body: formData,
      headers: { Accept: 'image/jpeg,image/png,image/webp' }, signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error('The local backend is offline. Start it and try again.')
  }
  if (!response.ok) throw new Error(await getErrorMessage(response))
  const rawMetadata = response.headers.get('X-Protection-Metadata')
  if (!rawMetadata) throw new Error('The backend returned an image without protection details.')
  let protection
  try {
    protection = JSON.parse(decodeURIComponent(rawMetadata))
  } catch {
    throw new Error('The backend returned invalid protection details.')
  }
  return { blob: await response.blob(), protection }
}

async function getVideoError(response) {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
  } catch {
    // Fall back to a stable local-processing message.
  }
  if (response.status === 413) return 'This video exceeds the configured size, duration, or resolution limit.'
  if (response.status === 415) return 'Choose a supported MP4, MOV, AVI or WEBM video.'
  return `Video request failed with status ${response.status}.`
}

export async function uploadVideo(file, signal) {
  const body = new FormData()
  body.append('video', file, file.name)
  let response
  try {
    response = await fetch(`${API_BASE_URL}/video/upload`, {
      method: 'POST', body, headers: { Accept: 'application/json' }, signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error('The local video service is offline.')
  }
  if (!response.ok) throw new Error(await getVideoError(response))
  return response.json()
}

export async function discardVideoUpload(uploadId) {
  if (!uploadId) return
  await fetch(`${API_BASE_URL}/video/upload/${encodeURIComponent(uploadId)}`, { method: 'DELETE' }).catch(() => {})
}

export async function startVideoProcessing(uploadId, settings, signal) {
  const response = await fetch(`${API_BASE_URL}/video/process`, {
    method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ upload_id: uploadId, settings }), signal,
  })
  if (!response.ok) throw new Error(await getVideoError(response))
  return response.json()
}

export async function getVideoJobStatus(jobId, signal) {
  const response = await fetch(`${API_BASE_URL}/video/status/${encodeURIComponent(jobId)}`, {
    headers: { Accept: 'application/json' }, signal,
  })
  if (!response.ok) throw new Error(await getVideoError(response))
  return response.json()
}

export async function cancelVideoJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/video/cancel/${encodeURIComponent(jobId)}`, {
    method: 'DELETE', headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw new Error(await getVideoError(response))
  return response.json()
}

export function getVideoResultUrl(jobId) {
  return `${API_BASE_URL}/video/result/${encodeURIComponent(jobId)}`
}
