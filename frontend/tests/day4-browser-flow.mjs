import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'

const targets = await fetch('http://127.0.0.1:9223/json').then((response) => response.json())
const page = targets.find((target) => target.type === 'page' && target.url.startsWith('http://127.0.0.1:5173'))
assert.ok(page, 'Application page not found')
const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
let id = 0
const pending = new Map()
socket.addEventListener('message', (event) => { const message = JSON.parse(event.data); if (pending.has(message.id)) { pending.get(message.id)(message.result); pending.delete(message.id) } })
const send = (method, params = {}) => new Promise((resolve) => { const callId = ++id; pending.set(callId, resolve); socket.send(JSON.stringify({ id: callId, method, params })) })
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result.value
const waitFor = async (expression, timeout = 120000) => { const start = Date.now(); while (Date.now() - start < timeout) { if (await evaluate(expression)) return; await new Promise((resolve) => setTimeout(resolve, 250)) } throw new Error(`Timed out: ${expression}`) }

await send('Runtime.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })
await send('Page.reload', { ignoreCache: true })
await waitFor(`document.body.innerText.includes('Backend connected')`)
const bytes = await readFile(`${process.env.TEMP}\\day4-bus.jpg`)
const base64 = bytes.toString('base64')
await evaluate(`(() => { const bytes=Uint8Array.from(atob('${base64}'), c=>c.charCodeAt(0)); const file=new File([bytes],'street.jpg',{type:'image/jpeg'}); const dt=new DataTransfer(); dt.items.add(file); const input=document.querySelector('input[aria-label="Choose a JPG, PNG or WEBP image"]'); Object.defineProperty(input,'files',{value:dt.files}); input.dispatchEvent(new Event('change',{bubbles:true})) })()`)
await waitFor(`document.querySelector('img[alt="Preview of street.jpg"]')?.naturalWidth > 0`)
await evaluate(`Array.from(document.querySelectorAll('button')).find(b=>b.innerText.includes('Analyze Privacy')).click()`)
await waitFor(`document.body.innerText.includes('DETECTION #1')`)
assert.equal(await evaluate(`document.body.innerText.includes('Bus') && document.body.innerText.includes('Person')`), true)
const boxesInside = `Array.from(document.querySelectorAll('div.border-cyan-400')).every(box=>{const b=box.getBoundingClientRect(),i=document.querySelector('img[alt="Preview of street.jpg"]').getBoundingClientRect();return b.left>=i.left-1&&b.top>=i.top-1&&b.right<=i.right+1&&b.bottom<=i.bottom+1&&b.width>0&&b.height>0})`
assert.equal(await evaluate(boxesInside), true, 'Desktop boxes must align within image')
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true })
assert.equal(await evaluate(boxesInside), true, 'Mobile boxes must align within image')
const count = await evaluate(`document.querySelectorAll('div.border-cyan-400').length`)
await evaluate(`document.querySelector('input[type="checkbox"]').click()`)
assert.equal(await evaluate(`document.querySelectorAll('div.border-cyan-400').length`), 0)
await evaluate(`document.querySelector('input[type="checkbox"]').click()`)
assert.equal(await evaluate(`document.querySelectorAll('div.border-cyan-400').length`), count)
const clip = await evaluate(`(() => { const r=document.querySelector('img[alt="Preview of street.jpg"]').getBoundingClientRect(); return {x:r.left,y:r.top+window.scrollY,width:r.width,height:r.height,scale:1} })()`)
const screenshot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip })
await writeFile('../docs/day4-mobile-detections.png', Buffer.from(screenshot.data, 'base64'))
socket.close()
console.log(`Day 4 browser flow passed with ${count} responsive bounding boxes.`)
