export const MAX_IMAGE_SIZE = 10 * 1024 * 1024
export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']

export function getImageFileError(file) {
  if (!file) return 'No image selected.'
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) return 'Unsupported image format. Choose a JPG, PNG or WEBP image.'
  if (file.size > MAX_IMAGE_SIZE) return 'Image must be smaller than 10 MB.'
  return ''
}

export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatFileType(type) {
  const labels = { 'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP' }
  return labels[type] || type
}
