const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/+$/, '')

async function apiRequest(path, options = {}) {
  const url = `${API_BASE_URL}${path}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  const data = await response.json().catch(() => null)

  if (!response.ok) {
    if (data?.details && Array.isArray(data.details)) {
      const msg = data.details
        .map((d) => (d.field ? `${d.field}: ${d.message}` : d.message))
        .join(' | ')
      throw new Error(msg || data.error || `Request failed (${response.status})`)
    }
    throw new Error(
      data?.detail || data?.error || `Request failed with status ${response.status}`
    )
  }

  return data
}

export function checkHealth() {
  return apiRequest('/api/health')
}

export function fetchFeatures() {
  return apiRequest('/api/features')
}

export function fetchMetrics() {
  return apiRequest('/api/metrics')
}

export function predictRisk(payload) {
  return apiRequest('/api/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export { API_BASE_URL }

