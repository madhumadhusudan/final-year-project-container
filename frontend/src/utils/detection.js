export function clamp(value, minimum, maximum) {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? Math.max(minimum, Math.min(maximum, numericValue)) : minimum
}

export function getBoundingBoxStyle(boundingBox, imageWidth, imageHeight) {
  if (!(imageWidth > 0) || !(imageHeight > 0)) return { display: 'none' }
  const x1 = clamp(boundingBox?.x1, 0, imageWidth)
  const y1 = clamp(boundingBox?.y1, 0, imageHeight)
  const x2 = clamp(boundingBox?.x2, x1, imageWidth)
  const y2 = clamp(boundingBox?.y2, y1, imageHeight)
  return {
    left: `${(x1 / imageWidth) * 100}%`,
    top: `${(y1 / imageHeight) * 100}%`,
    width: `${((x2 - x1) / imageWidth) * 100}%`,
    height: `${((y2 - y1) / imageHeight) * 100}%`,
  }
}

export function formatConfidence(confidence) {
  return `${(clamp(confidence, 0, 1) * 100).toFixed(1)}%`
}

export function formatClassName(className) {
  if (!className) return 'Object'
  return className.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}
