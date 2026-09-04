import test from 'node:test'
import assert from 'node:assert/strict'

import {
  RegionTracker, analysisToVideoCoordinates, calculateLiveRisk,
  cameraFailureState, canvasToVideoCoordinates, intersectionOverUnion, isFreshFrameResponse,
  modulesForRequest, shouldProtectTrack, stopMediaStream,
  videoToCanvasCoordinates,
} from '../src/utils/livePrivacy.js'

test('coordinate helpers map analysis, video, and canvas spaces consistently', () => {
  const video = analysisToVideoCoordinates(
    { x1: 64, y1: 48, x2: 320, y2: 240 },
    { width: 640, height: 480 }, { width: 1280, height: 960 },
  )
  assert.deepEqual(video, { x1: 128, y1: 96, x2: 640, y2: 480 })
  const canvas = videoToCanvasCoordinates(video, { width: 1280, height: 960 }, { width: 640, height: 480 })
  assert.deepEqual(canvas, { x1: 64, y1: 48, x2: 320, y2: 240 })
  assert.deepEqual(canvasToVideoCoordinates({ x: 320, y: 240 }, { width: 640, height: 480 }, { width: 1280, height: 960 }), { x: 640, y: 480 })
})

test('tracker associates regions, smooths motion, and keeps a miss for the grace period', () => {
  const tracker = new RegionTracker({ smoothingAlpha: 0.5, gracePeriodMs: 500 })
  const first = tracker.update([{ detection_id: 'face_1', category: 'face', confidence: 0.9, role: 'background_face', bounding_box: { x1: 10, y1: 10, x2: 50, y2: 50 } }], ['faces'], 1000)
  const trackId = first[0].trackId
  const moved = tracker.update([{ detection_id: 'face_1', category: 'face', confidence: 0.91, role: 'background_face', bounding_box: { x1: 20, y1: 10, x2: 60, y2: 50 } }], ['faces'], 1100)
  assert.equal(moved[0].trackId, trackId)
  assert.equal(moved[0].box.x1, 15)
  assert.equal(tracker.update([], ['faces'], 1200).length, 1)
  assert.equal(tracker.active(1699).length, 1)
  assert.equal(tracker.active(1701).length, 0)
})

test('unrefreshed slow categories survive fast face updates', () => {
  const tracker = new RegionTracker()
  tracker.update([{ detection_id: 'qr_1', category: 'qr_code', confidence: null, bounding_box: { x1: 5, y1: 5, x2: 25, y2: 25 } }], ['codes'], 100)
  const tracks = tracker.update([], ['faces'], 300)
  assert.equal(tracks.length, 1)
  assert.equal(tracks[0].category, 'qr_code')
})

test('stale geometry helpers and live risk use actual non-main tracks', () => {
  assert.equal(intersectionOverUnion({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 5, x2: 15, y2: 15 }), 25 / 175)
  const tracks = [
    { trackId: 'face_1', category: 'face', confidence: 1 },
    { trackId: 'face_2', category: 'face', confidence: 1 },
    { trackId: 'plate_1', category: 'license_plate', confidence: 1 },
  ]
  assert.deepEqual(calculateLiveRisk(tracks, true, 'face_1'), { score: 30, level: 'MODERATE' })
})

test('multi-rate scheduler adds at most one due heavy detector to faces', () => {
  const profile = { faces: 180, plates: 1200, cards: 1800, documents: 2000, codes: 1000, ocr: 1800 }
  const capabilities = { license_plates: true, payment_cards: true, identity_documents: true, qr_codes: true, barcodes: true, sensitive_text: true }
  const modules = modulesForRequest(2500, { codes: 2000, plates: 0, cards: 1000, documents: 1000, ocr: 1000 }, profile, capabilities)
  assert.deepEqual(modules, ['faces', 'plates'])
  assert.ok(modules.length <= 2)
})

test('stale responses are rejected and newer frame IDs are accepted', () => {
  assert.equal(isFreshFrameResponse(8, 7), true)
  assert.equal(isFreshFrameResponse(7, 7), false)
  assert.equal(isFreshFrameResponse(6, 7), false)
})

test('camera failures have explicit states and stream cleanup stops every track', () => {
  assert.equal(cameraFailureState('NotAllowedError').state, 'permission_denied')
  assert.equal(cameraFailureState('NotFoundError').state, 'camera_unavailable')
  assert.equal(cameraFailureState('NotReadableError').state, 'error')
  let stopped = 0
  const stream = { getTracks: () => [{ stop: () => { stopped += 1 } }, { stop: () => { stopped += 1 } }] }
  assert.equal(stopMediaStream(stream), 2)
  assert.equal(stopped, 2)
})

test('protection toggles preserve only the selected main face', () => {
  const settings = { enabled: { background_faces: true, qr_codes: false }, preserveMainSubject: true }
  assert.equal(shouldProtectTrack({ category: 'face', trackId: 'face_1' }, settings, 'face_1'), false)
  assert.equal(shouldProtectTrack({ category: 'face', trackId: 'face_2' }, settings, 'face_1'), true)
  assert.equal(shouldProtectTrack({ category: 'qr_code', trackId: 'qr_1' }, settings, 'face_1'), false)
})
