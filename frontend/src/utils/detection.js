export function clampUnit(value) {
  return Math.max(0, Math.min(1, Number(value) || 0))
}

export function getBoundingBoxStyle(normalizedBoundingBox) {
  const x = clampUnit(normalizedBoundingBox?.x)
  const y = clampUnit(normalizedBoundingBox?.y)
  const width = Math.min(clampUnit(normalizedBoundingBox?.width), 1 - x)
  const height = Math.min(clampUnit(normalizedBoundingBox?.height), 1 - y)
  return {
    left: `${x * 100}%`,
    top: `${y * 100}%`,
    width: `${width * 100}%`,
    height: `${height * 100}%`,
  }
}

export function formatConfidence(confidence) {
  return `${(clampUnit(confidence) * 100).toFixed(1)}%`
}

export function formatCategory(category) {
  return {
    face: 'Face',
    person: 'Person',
    background_person: 'Background person',
  }[category] || 'Detection'
}
