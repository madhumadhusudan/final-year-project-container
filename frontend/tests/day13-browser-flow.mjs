import assert from 'node:assert/strict'

const targets = await fetch('http://127.0.0.1:9223/json').then((response) => response.json())
const page = targets.find((target) => target.type === 'page' && target.url.startsWith('http://127.0.0.1:5173'))
assert.ok(page, 'Live privacy browser page was not found')
const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})

let id = 0
const pending = new Map()
const errors = []
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data)
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message.result)
    pending.delete(message.id)
  }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text)
  if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push(message.params.entry.text)
})
const send = (method, params = {}) => new Promise((resolve) => {
  const callId = ++id
  pending.set(callId, resolve)
  socket.send(JSON.stringify({ id: callId, method, params }))
})
const evaluate = async (expression) => {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
  return result.result.value
}
const waitFor = async (expression, label, timeout = 15000) => {
  const started = Date.now()
  while (Date.now() - started < timeout) {
    if (await evaluate(expression)) return
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  throw new Error(`Timed out: ${label}`)
}

await send('Runtime.enable')
await send('Log.enable')
await send('Page.enable')
await send('Page.addScriptToEvaluateOnNewDocument', { source: `
  (() => {
    window.__cameraRequests = [];
    window.__tracksStopped = 0;
    window.__analysisRequests = 0;
    const originalFetch = window.fetch.bind(window);
    window.fetch = (...args) => {
      if (String(args[0]).includes('/analyze-frame')) window.__analysisRequests += 1;
      return originalFetch(...args);
    };
    const makeStream = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 320; canvas.height = 240;
      const context = canvas.getContext('2d');
      context.fillStyle = '#0f766e'; context.fillRect(0, 0, 320, 240);
      context.fillStyle = '#fff'; context.fillRect(120, 70, 80, 100);
      const stream = canvas.captureStream(15);
      for (const track of stream.getTracks()) {
        const stop = track.stop.bind(track);
        track.stop = () => { window.__tracksStopped += 1; stop(); };
      }
      return stream;
    };
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getUserMedia: async (constraints) => { window.__cameraRequests.push(constraints); return makeStream(); },
      enumerateDevices: async () => [
        { kind: 'videoinput', deviceId: 'front', label: 'Synthetic front camera' },
        { kind: 'videoinput', deviceId: 'back', label: 'Synthetic back camera' },
      ],
    }});
  })();
` })
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true })
await send('Page.navigate', { url: 'http://127.0.0.1:5173/live' })
await waitFor(`document.body.innerText.includes('Live Privacy Protection')`, 'live page render')

for (const width of [375, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width, height: width === 375 ? 844 : 1000, deviceScaleFactor: 1, mobile: width === 375 })
  assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true, `${width}px layout overflowed`)
}
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true })

await evaluate(`Array.from(document.querySelectorAll('button')).find((button) => button.innerText.trim() === 'Start Camera').click()`)
await waitFor(`document.body.innerText.includes('LIVE · Protected preview') && document.querySelector('video').srcObject !== null`, 'camera active')
assert.equal(await evaluate(`window.__cameraRequests[0].audio === false && Boolean(window.__cameraRequests[0].video)`), true, 'camera must request video only')
assert.equal(await evaluate(`getComputedStyle(document.querySelector('video')).opacity === '0'`), true, 'raw video must stay hidden')
await waitFor(`window.__analysisRequests > 0`, 'live analysis request')
await waitFor(`document.body.innerText.includes('Detection') && document.body.innerText.includes('Active')`, 'active detection')

await evaluate(`Array.from(document.querySelectorAll('button')).find((button) => button.innerText.trim() === 'Stop Camera').click()`)
await waitFor(`document.querySelector('video').srcObject === null && document.body.innerText.includes('Camera stopped')`, 'camera stop cleanup')
assert.ok(await evaluate(`window.__tracksStopped`), 'media track stop was not called')
const stoppedRequestCount = await evaluate(`window.__analysisRequests`)
await new Promise((resolve) => setTimeout(resolve, 900))
assert.equal(await evaluate(`window.__analysisRequests`), stoppedRequestCount, 'analysis continued after camera stop')

await evaluate(`Array.from(document.querySelectorAll('button')).find((button) => button.innerText.trim() === 'Start Camera').click()`)
await waitFor(`document.querySelector('video').srcObject !== null`, 'camera restarted')
const stoppedBeforeNavigation = await evaluate(`window.__tracksStopped`)
await evaluate(`Array.from(document.querySelectorAll('a')).find((link) => link.innerText.trim() === 'Image Privacy').click()`)
await waitFor(`document.body.innerText.includes('Check an image before it goes public')`, 'SPA navigation to image mode')
assert.ok(await evaluate(`window.__tracksStopped`) > stoppedBeforeNavigation, 'navigation did not stop the live media track')

assert.deepEqual(errors, [], `Browser errors: ${errors.join('; ')}`)
socket.close()
console.log('Day 13 browser flow passed: responsive /live route, video-only start, hidden raw source, analysis, stop cleanup, and navigation cleanup.')
