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

