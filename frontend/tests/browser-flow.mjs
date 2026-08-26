import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'

const targets = await fetch('http://127.0.0.1:9223/json').then((response) => response.json())
const page = targets.find((target) => target.type === 'page' && target.url.startsWith('http://127.0.0.1:5173'))
assert.ok(page, 'Local application page was not found in Edge')

const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})

let nextId = 1
const pending = new Map()
const browserErrors = []

socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data)
  if (message.id && pending.has(message.id)) {
    const { resolve, reject, method } = pending.get(message.id)
    pending.delete(message.id)
    if (message.error) reject(new Error(`${method}: ${message.error.message}`))
    else resolve(message.result)
  }
  if (message.method === 'Runtime.exceptionThrown') browserErrors.push(message.params.exceptionDetails.text)
  if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') browserErrors.push(message.params.entry.text)
})

function send(method, params = {}) {
  const id = nextId++
  socket.send(JSON.stringify({ id, method, params }))
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }))
}

async function evaluate(expression) {
  const response = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text)
  return response.result.value
}

async function waitFor(expression, message, timeout = 5000) {
  const started = Date.now()
  while (Date.now() - started < timeout) {
    try {
      if (await evaluate(expression)) return
    } catch {
      // A navigation can briefly replace the page's JavaScript context.
    }
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  throw new Error(`Timed out: ${message}`)
}

const selectGeneratedImage = (width, height, type, name, mode = 'change') => `
  (() => {
    const canvas = document.createElement('canvas');
    canvas.width = ${width}; canvas.height = ${height};
    const context = canvas.getContext('2d');
    context.fillStyle = '#0f766e'; context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#ffffff'; context.font = '24px sans-serif'; context.fillText('${width} x ${height}', 20, 40);
    const encoded = canvas.toDataURL('${type}', 0.9).split(',')[1];
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const file = new File([bytes], '${name}', { type: '${type}' });
    const transfer = new DataTransfer(); transfer.items.add(file);
    const input = document.querySelector('${mode === 'replace' ? 'input[aria-label="Choose a replacement image"]' : 'input[aria-label="Choose a JPG, PNG or WEBP image"]'}');
    Object.defineProperty(input, 'files', { configurable: true, value: transfer.files });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return file.size;
  })()
`

const chooseInvalidFile = (type, name, size) => `
  (() => {
    const file = new File([new Uint8Array(${size})], '${name}', { type: '${type}' });
    const transfer = new DataTransfer(); transfer.items.add(file);
    const input = document.querySelector('input[aria-label="Choose a replacement image"]');
    Object.defineProperty(input, 'files', { configurable: true, value: transfer.files });
    input.dispatchEvent(new Event('change', { bubbles: true }));
  })()
`

await send('Runtime.enable')
await send('Log.enable')
await send('Network.enable')
await send('Network.setBlockedURLs', { urls: [] })
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })
await send('Page.reload', { ignoreCache: true })
await waitFor(`document.readyState === 'complete'`, 'initial page load')
await waitFor(`document.body.innerText.includes('Backend connected')`, 'real backend connection')

assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true, 'Desktop page must not overflow horizontally')
assert.equal(await evaluate(`document.body.innerText.includes('Drop your image here')`), true)
assert.equal(await evaluate(`document.body.innerText.includes('No analysis results yet')`), true)

await send('Emulation.setDeviceMetricsOverride', { width: 375, height: 844, deviceScaleFactor: 1, mobile: true })
assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true, 'Mobile page must not overflow horizontally')
assert.equal(await evaluate(`Boolean(document.querySelector('button[aria-label="Open navigation menu"]'))`), true)
const mobileScreenshot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true })
await writeFile('../docs/day2-mobile-emulated.png', Buffer.from(mobileScreenshot.data, 'base64'))
await evaluate(`document.querySelector('button[aria-label="Open navigation menu"]').click()`)
assert.equal(await evaluate(`document.body.innerText.includes('How It Works')`), true)
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })

await evaluate(selectGeneratedImage(400, 700, 'image/jpeg', 'portrait-photo.jpg'))
await waitFor(`document.body.innerText.includes('400 × 700')`, 'portrait JPG metadata')
assert.equal(await evaluate(`document.body.innerText.includes('portrait-photo.jpg') && document.body.innerText.includes('JPEG')`), true)
assert.equal(await evaluate(`document.querySelector('img[alt="Preview of portrait-photo.jpg"]').naturalWidth === 400`), true)

await evaluate(`Array.from(document.querySelectorAll('button')).find(button => button.innerText.includes('Analyze Privacy')).click()`)
await waitFor(`document.body.innerText.includes('AI detection will be connected in the next development stage.')`, 'analysis readiness state')
assert.equal(await evaluate(`document.body.innerText.includes('Ready for Day 3 AI integration')`), true)

await evaluate(selectGeneratedImage(600, 600, 'image/png', 'square-photo.png', 'replace'))
await waitFor(`document.body.innerText.includes('600 × 600')`, 'square PNG replacement')
assert.equal(await evaluate(`document.body.innerText.includes('square-photo.png') && document.body.innerText.includes('PNG')`), true)

await evaluate(chooseInvalidFile('application/pdf', 'private.pdf', 128))
await waitFor(`document.body.innerText.includes('Unsupported image format.')`, 'invalid PDF validation')
assert.equal(await evaluate(`document.body.innerText.includes('square-photo.png')`), true, 'Invalid replacement should preserve current image')

await evaluate(chooseInvalidFile('image/jpeg', 'too-large.jpg', 10 * 1024 * 1024 + 1))
await waitFor(`document.body.innerText.includes('Image must be smaller than 10 MB.')`, 'oversize validation')

await evaluate(selectGeneratedImage(800, 400, 'image/webp', 'landscape-photo.webp', 'replace'))
await waitFor(`document.body.innerText.includes('800 × 400')`, 'landscape WEBP replacement')
assert.equal(await evaluate(`document.body.innerText.includes('WEBP')`), true)

await evaluate(`Array.from(document.querySelectorAll('button')).find(button => button.innerText.includes('Remove image')).click()`)
await waitFor(`document.body.innerText.includes('Drop your image here')`, 'remove resets uploader')
assert.equal(await evaluate(`document.querySelector('img[alt^="Preview of"]') === null`), true)

await evaluate(`
  (() => {
    const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 180;
    const encoded = canvas.toDataURL('image/png').split(',')[1];
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const file = new File([bytes], 'dropped-photo.png', { type: 'image/png' });
    const transfer = new DataTransfer(); transfer.items.add(file);
    const dropZone = Array.from(document.querySelectorAll('button')).find(button => button.innerText.includes('Drop your image here'));
    dropZone.dispatchEvent(new DragEvent('dragenter', { bubbles: true, dataTransfer: transfer }));
    dropZone.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: transfer }));
  })()
`)
await waitFor(`document.body.innerText.includes('320 × 180')`, 'drag and drop image selection')
await evaluate(`Array.from(document.querySelectorAll('button')).find(button => button.innerText.includes('Remove image')).click()`)
await waitFor(`document.body.innerText.includes('Drop your image here')`, 'second remove')
await evaluate(selectGeneratedImage(320, 180, 'image/png', 'dropped-photo.png'))
await waitFor(`document.body.innerText.includes('320 × 180')`, 'repeated image selection')

await send('Network.setBlockedURLs', { urls: ['*://127.0.0.1:8000/*'] })
await send('Page.reload', { ignoreCache: true })
await waitFor(`document.body.innerText.includes('Backend offline')`, 'offline backend state')
assert.equal(await evaluate(`document.body.innerText.includes('Drop your image here')`), true, 'Frontend must remain usable while backend is offline')
assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true)

await send('Network.setBlockedURLs', { urls: [] })
assert.deepEqual(browserErrors, [], `Browser errors detected: ${browserErrors.join('; ')}`)
socket.close()

console.log('Browser flow passed: connected/offline states, responsive layouts, browse/drop, JPG/PNG/WEBP metadata, validation, analyze, replace, remove, and repeated upload.')
