import test from 'node:test'
import assert from 'node:assert/strict'
import { clampUnit, formatCategory, formatConfidence, getBoundingBoxStyle } from '../src/utils/detection.js'

test('maps normalized model coordinates to responsive percentage styles', () => {
  assert.deepEqual(
    getBoundingBoxStyle({ x: 0.125, y: 0.25, width: 0.5, height: 0.4 }),
    { left: '12.5%', top: '25%', width: '50%', height: '40%' },
  )
})

test('clamps boxes to the rendered image boundary', () => {
  const style = getBoundingBoxStyle({ x: 0.9, y: -0.2, width: 0.4, height: 2 })
  assert.equal(style.left, '90%')
  assert.equal(style.top, '0%')
  assert.ok(Math.abs(Number.parseFloat(style.width) - 10) < 0.0001)
  assert.equal(style.height, '100%')
  assert.equal(clampUnit(Number.NaN), 0)
})

test('formats confidence and supported categories for the results UI', () => {
  assert.equal(formatConfidence(0.9743), '97.4%')
  assert.equal(formatCategory('background_person'), 'Background person')
})
