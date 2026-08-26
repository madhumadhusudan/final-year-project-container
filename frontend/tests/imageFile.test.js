import test from 'node:test'
import assert from 'node:assert/strict'
import { MAX_IMAGE_SIZE, formatFileSize, formatFileType, getImageFileError } from '../src/utils/imageFile.js'

function mockFile({ name = 'photo.jpg', type = 'image/jpeg', size = 1024 } = {}) {
  return { name, type, size }
}

test('accepts JPG, PNG and WEBP image MIME types', () => {
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) {
    assert.equal(getImageFileError(mockFile({ type })), '')
  }
})

test('rejects PDF, text and untyped files', () => {
  for (const type of ['application/pdf', 'text/plain', '']) {
    assert.match(getImageFileError(mockFile({ type })), /Unsupported image format/)
  }
})

test('accepts exactly 10 MB and rejects larger images', () => {
  assert.equal(getImageFileError(mockFile({ size: MAX_IMAGE_SIZE })), '')
  assert.equal(getImageFileError(mockFile({ size: MAX_IMAGE_SIZE + 1 })), 'Image must be smaller than 10 MB.')
})

test('handles a missing selection safely', () => {
  assert.equal(getImageFileError(), 'No image selected.')
})

test('formats file metadata for display', () => {
  assert.equal(formatFileSize(400), '400 B')
  assert.equal(formatFileSize(1536), '1.5 KB')
  assert.equal(formatFileSize(2.4 * 1024 * 1024), '2.4 MB')
  assert.equal(formatFileType('image/jpeg'), 'JPEG')
  assert.equal(formatFileType('image/png'), 'PNG')
  assert.equal(formatFileType('image/webp'), 'WEBP')
})
