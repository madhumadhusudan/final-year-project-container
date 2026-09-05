import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_MAX_VIDEO_BYTES, formatBytes, formatDuration, getVideoFileError } from '../src/utils/videoFile.js'

const video = (name, size = 1024) => ({ name, size })

test('video validation accepts supported extensions and configured size boundary', () => {
  for (const extension of ['mp4', 'mov', 'avi', 'webm']) {
    assert.equal(getVideoFileError(video(`safe.${extension}`)), '')
  }
  assert.equal(getVideoFileError(video('limit.mp4', DEFAULT_MAX_VIDEO_BYTES)), '')
})

test('video validation rejects missing, empty, oversized, and unsupported files', () => {
  assert.match(getVideoFileError(null), /Choose a video/)
  assert.match(getVideoFileError(video('empty.mp4', 0)), /empty/)
  assert.match(getVideoFileError(video('large.mp4', DEFAULT_MAX_VIDEO_BYTES + 1)), /150 MB/)
  assert.match(getVideoFileError(video('document.pdf')), /MP4/)
})

test('video metadata formatters use real values', () => {
  assert.equal(formatDuration(65.2), '1:05')
  assert.equal(formatBytes(2 * 1024 * 1024), '2.0 MB')
})
