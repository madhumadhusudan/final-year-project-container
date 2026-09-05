export const VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'webm']
export const DEFAULT_MAX_VIDEO_BYTES = 150 * 1024 * 1024

export function getVideoFileError(file, maxBytes = DEFAULT_MAX_VIDEO_BYTES) {
  if (!file) return 'Choose a video first.'
  const extension = file.name.split('.').pop()?.toLowerCase()
  if (!VIDEO_EXTENSIONS.includes(extension)) return 'Choose an MP4, MOV, AVI or WEBM video.'
  if (!file.size) return 'The selected video is empty.'
  if (file.size > maxBytes) return `Video must be ${Math.floor(maxBytes / 1024 / 1024)} MB or smaller.`
  return ''
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'Unavailable'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return 'Unavailable'
  const minutes = Math.floor(seconds / 60)
  const remaining = Math.round(seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${remaining}`
}
