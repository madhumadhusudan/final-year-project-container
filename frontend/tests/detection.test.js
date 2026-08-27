import test from 'node:test'
import assert from 'node:assert/strict'
import { clamp, formatClassName, formatConfidence, getBoundingBoxStyle } from '../src/utils/detection.js'

test('maps original pixel coordinates to responsive percentage styles', () => {
  assert.deepEqual(getBoundingBoxStyle({ x1: 240, y1: 270, x2: 1200, y2: 702 }, 1920, 1080), {
    left: '12.5%', top: '25%', width: '50%', height: '40%',
  })
})

test('clamps invalid boxes to image bounds without NaN or negative dimensions', () => {
  const style = getBoundingBoxStyle({ x1: 900, y1: -20, x2: 1400, y2: 2000 }, 1000, 1000)
  assert.equal(style.left, '90%')
  assert.equal(style.top, '0%')
  assert.ok(Math.abs(Number.parseFloat(style.width) - 10) < 0.0001)
  assert.equal(style.height, '100%')
  assert.equal(clamp(Number.NaN, 0, 1), 0)
})

test('formats real class names and confidence', () => {
  assert.equal(formatConfidence(0.9743), '97.4%')
  assert.equal(formatClassName('traffic_light'), 'Traffic Light')
})
