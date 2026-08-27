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

export async function analyzeImage(file, signal) {
  const formData = new FormData()
  formData.append('image', file, file.name)
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
    headers: { Accept: 'application/json' },
    signal,
  })

  if (!response.ok) throw new Error(await getErrorMessage(response))
  return response.json()
}
