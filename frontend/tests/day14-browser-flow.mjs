import assert from 'node:assert/strict'

const targets = await fetch('http://127.0.0.1:9224/json').then((response) => response.json())
const page = targets.find((target) => target.type === 'page' && target.url.includes('127.0.0.1:5173'))
assert.ok(page, 'Video privacy browser page was not found')
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
const waitFor = async (expression, label, timeout = 10000) => {
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
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => String(input).endsWith('/health')
      ? Promise.resolve(new Response(JSON.stringify({ status: 'healthy' }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      : originalFetch(input, init);
  })();
` })
errors.length = 0
await send('Page.navigate', { url: 'http://127.0.0.1:5173/video' })
await waitFor(`document.body.innerText.includes('Video Privacy Protection')`, 'video page render')

for (const width of [375, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width, height: width === 375 ? 844 : 1000, deviceScaleFactor: 1, mobile: width === 375 })
  assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true, `${width}px layout overflowed`)
}

assert.equal(await evaluate(`document.querySelector('input[type=file]').accept.includes('.mp4')`), true)
assert.equal(await evaluate(`Array.from(document.querySelectorAll('button')).find((button) => button.innerText.includes('Start privacy analysis')).disabled`), true)
assert.equal(await evaluate(`document.body.innerText.includes('Preserve Main Subject') && document.body.innerText.includes('Quality profile')`), true)
assert.deepEqual(errors, [], `Browser errors: ${errors.join('; ')}`)
socket.close()
console.log('Day 14 browser flow passed: /video route, responsive layout, upload contract, privacy settings, and safe initial state.')
