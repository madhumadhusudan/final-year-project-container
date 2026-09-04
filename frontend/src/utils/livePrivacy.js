import { clamp } from './detection.js'

export const CATEGORY_MODULES = {
  face: 'faces',
  license_plate: 'plates',
  payment_card: 'cards',
  identity_document: 'documents',
  qr_code: 'codes',
  barcode: 'codes',
  sensitive_text: 'ocr',
}

export const MODULE_CATEGORIES = {
  faces: ['face'],
  plates: ['license_plate'],
  cards: ['payment_card'],
  documents: ['identity_document'],
  codes: ['qr_code', 'barcode'],
  ocr: ['sensitive_text'],
}

export function analysisToVideoCoordinates(box, analysisSize, videoSize) {
  const scaleX = videoSize.width / Math.max(1, analysisSize.width)
  const scaleY = videoSize.height / Math.max(1, analysisSize.height)
  return {
    x1: clamp(box.x1 * scaleX, 0, videoSize.width),
    y1: clamp(box.y1 * scaleY, 0, videoSize.height),
    x2: clamp(box.x2 * scaleX, 0, videoSize.width),
    y2: clamp(box.y2 * scaleY, 0, videoSize.height),
  }
}

export function videoToCanvasCoordinates(box, videoSize, canvasSize) {
  const scaleX = canvasSize.width / Math.max(1, videoSize.width)
  const scaleY = canvasSize.height / Math.max(1, videoSize.height)
  return {
    x1: clamp(box.x1 * scaleX, 0, canvasSize.width),
    y1: clamp(box.y1 * scaleY, 0, canvasSize.height),
    x2: clamp(box.x2 * scaleX, 0, canvasSize.width),
    y2: clamp(box.y2 * scaleY, 0, canvasSize.height),
  }
}

export function canvasToVideoCoordinates(point, canvasSize, videoSize) {
  return {
    x: clamp(point.x * videoSize.width / Math.max(1, canvasSize.width), 0, videoSize.width),
    y: clamp(point.y * videoSize.height / Math.max(1, canvasSize.height), 0, videoSize.height),
  }
}

export function intersectionOverUnion(first, second) {
  const x1 = Math.max(first.x1, second.x1)
  const y1 = Math.max(first.y1, second.y1)
  const x2 = Math.min(first.x2, second.x2)
  const y2 = Math.min(first.y2, second.y2)
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (!intersection) return 0
  const firstArea = Math.max(0, first.x2 - first.x1) * Math.max(0, first.y2 - first.y1)
  const secondArea = Math.max(0, second.x2 - second.x1) * Math.max(0, second.y2 - second.y1)
  return intersection / Math.max(1, firstArea + secondArea - intersection)
}

function centerDistance(first, second) {
  const firstX = (first.x1 + first.x2) / 2
  const firstY = (first.y1 + first.y2) / 2
  const secondX = (second.x1 + second.x2) / 2
  const secondY = (second.y1 + second.y2) / 2
  const distance = Math.hypot(firstX - secondX, firstY - secondY)
  const reference = Math.max(1, Math.hypot(first.x2 - first.x1, first.y2 - first.y1))
  return distance / reference
}

export function smoothBox(previous, current, alpha = 0.68) {
  const weight = clamp(alpha, 0, 1)
  return Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map((key) => [
    key, previous[key] * (1 - weight) + current[key] * weight,
  ]))
}

const DEFAULT_TTL = {
  face: 700,
  license_plate: 2800,
  payment_card: 3400,
  identity_document: 3600,
  qr_code: 2600,
  barcode: 2600,
  sensitive_text: 4000,
}

export class RegionTracker {
  constructor({ smoothingAlpha = 0.68, gracePeriodMs = 550, ttlByCategory = DEFAULT_TTL } = {}) {
    this.smoothingAlpha = smoothingAlpha
    this.gracePeriodMs = gracePeriodMs
    this.ttlByCategory = { ...DEFAULT_TTL, ...ttlByCategory }
    this.tracks = []
    this.nextId = 1
  }

  clear() {
    this.tracks = []
    this.nextId = 1
  }

  active(now = performance.now()) {
    this.tracks = this.tracks.filter((track) => now <= track.expiresAt)
    return this.tracks
  }

  update(regions, refreshedModules, now = performance.now()) {
    const refreshed = new Set(refreshedModules.flatMap((name) => MODULE_CATEGORIES[name] || []))
    const candidates = this.active(now)
    const used = new Set()
    const next = []

    for (const region of regions) {
      let bestIndex = -1
      let bestScore = -Infinity
      candidates.forEach((track, index) => {
        if (used.has(index) || track.category !== region.category) return
        const overlap = intersectionOverUnion(track.box, region.bounding_box)
        const distance = centerDistance(track.box, region.bounding_box)
        if (overlap < 0.12 && distance > 0.85) return
        const score = overlap * 2 - distance
        if (score > bestScore) {
          bestScore = score
          bestIndex = index
        }
      })
      const matched = bestIndex >= 0 ? candidates[bestIndex] : null
      if (matched) used.add(bestIndex)
      const category = region.category
      next.push({
        ...(matched || {}),
        trackId: matched?.trackId || `${category}_${this.nextId++}`,
        category,
        box: matched ? smoothBox(matched.box, region.bounding_box, this.smoothingAlpha) : { ...region.bounding_box },
        confidence: region.confidence,
        role: region.role,
        sourceDetectionId: region.detection_id,
        lastSeen: now,
        expiresAt: now + this.ttlByCategory[category],
        missedAt: undefined,
      })
    }

    candidates.forEach((track, index) => {
      if (used.has(index) || next.some((item) => item.trackId === track.trackId)) return
      if (!refreshed.has(track.category)) {
        next.push(track)
      } else if (track.missedAt == null) {
        next.push({ ...track, missedAt: now, expiresAt: now + this.gracePeriodMs })
      } else if (now - track.missedAt <= this.gracePeriodMs) {
        next.push(track)
      }
    })
    this.tracks = next
    return this.active(now)
  }
}

export function paddedBox(box, category, canvasSize) {
  const ratios = {
    face: 0.1, license_plate: 0.08, payment_card: 0.06,
    identity_document: 0.04, qr_code: 0.14, barcode: 0.1, sensitive_text: 0.08,
  }
  const padding = Math.max(3, Math.min(box.x2 - box.x1, box.y2 - box.y1) * (ratios[category] ?? 0.06))
  return {
    x1: clamp(box.x1 - padding, 0, canvasSize.width),
    y1: clamp(box.y1 - padding, 0, canvasSize.height),
    x2: clamp(box.x2 + padding, 0, canvasSize.width),
    y2: clamp(box.y2 + padding, 0, canvasSize.height),
  }
}

export function calculateLiveRisk(tracks, preserveMainSubject, mainTrackId) {
  const weights = {
    face: 12, license_plate: 18, payment_card: 27, identity_document: 28,
    qr_code: 14, barcode: 5, sensitive_text: 12,
  }
  const caps = {
    face: 35, license_plate: 30, payment_card: 35, identity_document: 40,
    qr_code: 30, barcode: 12, sensitive_text: 35,
  }
  const totals = {}
  tracks.forEach((track) => {
    if (track.category === 'face' && preserveMainSubject && track.trackId === mainTrackId) return
    const confidence = track.confidence == null ? 0.8 : clamp(track.confidence, 0.45, 1)
    totals[track.category] = (totals[track.category] || 0) + (weights[track.category] || 0) * confidence
  })
  const score = Math.min(100, Math.round(Object.entries(totals).reduce(
    (sum, [category, value]) => sum + Math.min(value, caps[category] || value), 0,
  )))
  const level = score < 20 ? 'LOW' : score < 40 ? 'MODERATE' : score < 60 ? 'ELEVATED' : score < 80 ? 'HIGH' : 'CRITICAL'
  return { score, level }
}

export function modulesForRequest(now, lastRun, intervals, capabilities) {
  const candidates = [
    ['codes', ['qr_codes', 'barcodes']],
    ['plates', ['license_plates']],
    ['cards', ['payment_cards']],
    ['documents', ['identity_documents']],
    ['ocr', ['sensitive_text']],
  ]
  const due = candidates
    .filter(([module, categories]) => categories.some((category) => capabilities[category]) && now - (lastRun[module] || 0) >= intervals[module])
    .sort(([first], [second]) => ((lastRun[first] || 0) + intervals[first]) - ((lastRun[second] || 0) + intervals[second]))
  return ['faces', ...(due[0] ? [due[0][0]] : [])]
}

export function isFreshFrameResponse(frameId, lastAcceptedFrameId) {
  return Number.isInteger(frameId) && frameId > lastAcceptedFrameId
}

export function stopMediaStream(stream) {
  if (!stream?.getTracks) return 0
  const tracks = stream.getTracks()
  tracks.forEach((track) => track.stop())
  return tracks.length
}

export function cameraFailureState(errorName) {
  if (errorName === 'NotAllowedError' || errorName === 'SecurityError') {
    return { state: 'permission_denied', message: 'Camera permission was denied. Allow camera access in browser settings, then retry.' }
  }
  if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError') {
    return { state: 'camera_unavailable', message: 'No available camera was found on this device.' }
  }
  return { state: 'error', message: 'The camera could not start. It may already be in use by another application.' }
}

export function shouldProtectTrack(track, settings, mainTrackId) {
  const key = {
    face: 'background_faces', license_plate: 'license_plates', payment_card: 'payment_cards',
    identity_document: 'identity_documents', qr_code: 'qr_codes', barcode: 'barcodes',
    sensitive_text: 'sensitive_text',
  }[track.category]
  return Boolean(
    key && settings.enabled[key]
    && !(track.category === 'face' && settings.preserveMainSubject && track.trackId === mainTrackId),
  )
}
