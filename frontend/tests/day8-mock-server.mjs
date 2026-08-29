import http from 'node:http'

let lastImage = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64')

function extractPng(body) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const start = body.indexOf(signature)
  const marker = Buffer.from('IEND')
  const markerIndex = body.indexOf(marker, start)
  if (start >= 0 && markerIndex >= 0) return body.subarray(start, markerIndex + 8)
  return null
}

function jsonPart(body, name) {
  const text = body.toString('utf8')
  const marker = `name="${name}"`
  const start = text.indexOf(marker)
  if (start < 0) return {}
  const valueStart = text.indexOf('\r\n\r\n', start) + 4
  const valueEnd = text.indexOf('\r\n--', valueStart)
  try { return JSON.parse(text.slice(valueStart, valueEnd)) } catch { return {} }
}

const analysis = {
  status: 'success', image: { filename: 'day8-scene.png', width: 320, height: 180, format: 'PNG' },
  analysis: {
    status: 'completed', model: 'mock', detection_count: 1,
    detections: [{ id: 1, class_id: 0, class_name: 'person', confidence: 0.96, bounding_box: { x1: 20, y1: 10, x2: 150, y2: 175 } }],
    object_detection: { model: 'mock', detection_count: 1, detections: [] },
    face_detection: { status: 'completed', detector: 'mock', face_count: 2, faces: [
      { face_id: 1, confidence: 0.98, bounding_box: { x1: 45, y1: 25, x2: 110, y2: 90 }, width: 65, height: 65, area: 4225, area_ratio: 0.073, center: { x: 77.5, y: 57.5 }, normalized_center: { x: 0.24, y: 0.32 }, distance_from_image_center: 0.2, role: 'main_subject' },
      { face_id: 2, confidence: 0.91, bounding_box: { x1: 230, y1: 30, x2: 270, y2: 70 }, width: 40, height: 40, area: 1600, area_ratio: 0.028, center: { x: 250, y: 50 }, normalized_center: { x: 0.78, y: 0.28 }, distance_from_image_center: 0.3, role: 'background_face' },
    ] },
    main_subject: { status: 'identified', face_id: 1, subject_score: 0.91, reason: 'Mock context result.' },
    license_plate_detection: { status: 'completed', detector: 'mock', plate_count: 1, plates: [{ id: 1, class_name: 'license_plate', confidence: 0.9, bounding_box: { x1: 200, y1: 130, x2: 275, y2: 155 } }] },
    card_detection: { status: 'completed', detector: 'mock', card_count: 1, cards: [{ id: 1, class_name: 'card', confidence: 0.89, bounding_box: { x1: 120, y1: 100, x2: 190, y2: 145 } }] },
    ocr: { status: 'completed', engine: 'mock', languages: ['en'], text_count: 1, texts: [] },
    sensitive_text: { status: 'completed', count: 1, items: [{ id: 1, text_id: 1, type: 'email', masked_value: 't***@example.com', confidence: 0.94, reason: 'Mock sensitive text.', bounding_box: { x1: 10, y1: 150, x2: 105, y2: 170 } }] },
    privacy_risk: {
      score: 70, level: 'HIGH', summary: 'Multiple high-impact privacy-sensitive elements are visible.',
      breakdown: { background_faces: 12, license_plates: 20, payment_cards: 30, sensitive_text: 8, context_uncertainty: 0 },
      factors: [], top_risks: ['Payment card exposed', '1 license plate exposed', '1 background face visible'],
      recommendations: ['Hide the complete payment-card region.', 'Anonymize visible license plates.', 'Protect background faces.'],
      assessment: { status: 'complete', unavailable_modules: [] },
    },
  },
  performance: { inference_time_ms: 5, object_detection_ms: 5, face_detection_ms: 2, license_plate_detection_ms: 1, card_detection_ms: 1, context_analysis_ms: 1, ocr_detection_ms: 2, sensitive_text_analysis_ms: 1, total_analysis_ms: 13 },
}

const server = http.createServer((request, response) => {
  response.setHeader('Access-Control-Allow-Origin', 'http://127.0.0.1:5173')
  response.setHeader('Access-Control-Expose-Headers', 'X-Protection-Metadata, Content-Disposition')
  if (request.method === 'OPTIONS') {
    response.writeHead(204, { 'Access-Control-Allow-Methods': 'GET,POST,OPTIONS', 'Access-Control-Allow-Headers': 'Accept' })
    return response.end()
  }
  if (request.url === '/health') {
    response.writeHead(200, { 'Content-Type': 'application/json' })
    return response.end(JSON.stringify({ status: 'healthy', service: 'day8-browser-mock' }))
  }
  const chunks = []
  request.on('data', (chunk) => chunks.push(chunk))
  request.on('end', () => {
    const body = Buffer.concat(chunks)
    const image = extractPng(body)
    if (image) lastImage = image
    if (request.url === '/analyze') {
      response.writeHead(200, { 'Content-Type': 'application/json' })
      return response.end(JSON.stringify(analysis))
    }
    if (request.url === '/protect') {
      const settings = jsonPart(body, 'settings')
      const breakdown = {
        background_faces: settings.protect_background_faces === false ? 0 : 1,
        license_plates: settings.protect_license_plates === false ? 0 : 1,
        cards: settings.protect_cards === false ? 0 : 1,
        sensitive_text: settings.protect_sensitive_text === false ? 0 : 1,
      }
      const protection = {
        status: 'completed', method: settings.anonymization_method || 'blur', strength: settings.strength || 'medium',
        regions_protected: Object.values(breakdown).reduce((total, count) => total + count, 0), breakdown,
        main_subject_preserved: true, warnings: [],
        risk: {
          before: { score: 70, level: 'HIGH' },
          after: { score: settings.protect_license_plates === false ? 20 : 7, level: settings.protect_license_plates === false ? 'MODERATE' : 'LOW' },
          reduction: settings.protect_license_plates === false ? 50 : 63,
          reduction_percent: settings.protect_license_plates === false ? 71.4 : 90.0,
        },
      }
      response.writeHead(200, {
        'Content-Type': 'image/png', 'Cache-Control': 'no-store',
        'X-Protection-Metadata': encodeURIComponent(JSON.stringify(protection)),
      })
      return response.end(lastImage)
    }
    response.writeHead(404, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify({ detail: 'Not found' }))
  })
})

server.listen(8000, '127.0.0.1', () => console.log('Day 8 browser mock listening on 8000'))
