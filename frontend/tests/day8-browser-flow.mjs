import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'

const targets = await fetch('http://127.0.0.1:9223/json').then((response) => response.json())
const page = targets.find((target) => target.type === 'page' && target.url.startsWith('http://127.0.0.1:5173'))
assert.ok(page, 'Day 8 application page was not found')
const socket = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
let id = 0
const pending = new Map()
const errors = []
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data)
  if (message.id && pending.has(message.id)) { pending.get(message.id)(message.result); pending.delete(message.id) }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text)
  if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push(message.params.entry.text)
})
const send = (method, params = {}) => new Promise((resolve) => { const callId = ++id; pending.set(callId, resolve); socket.send(JSON.stringify({ id: callId, method, params })) })
const evaluate = async (expression) => { const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); if (result.exceptionDetails) throw new Error(result.exceptionDetails.text); return result.result.value }
const waitFor = async (expression, label, timeout = 10000) => { const started = Date.now(); while (Date.now() - started < timeout) { if (await evaluate(expression)) return; await new Promise((resolve) => setTimeout(resolve, 100)) } throw new Error(`Timed out: ${label}`) }

await send('Runtime.enable')
await send('Log.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })
await send('Page.reload', { ignoreCache: true })
await waitFor(`document.body.innerText.includes('Backend connected')`, 'backend connection')
for (const width of [375, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width, height: width === 375 ? 844 : 1000, deviceScaleFactor: 1, mobile: width === 375 })
  assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true, `${width}px layout overflowed`)
}
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })

await evaluate(`(() => { const canvas=document.createElement('canvas'); canvas.width=320; canvas.height=180; const c=canvas.getContext('2d'); const gradient=c.createLinearGradient(0,0,320,180); gradient.addColorStop(0,'#0f766e'); gradient.addColorStop(1,'#f59e0b'); c.fillStyle=gradient; c.fillRect(0,0,320,180); const bytes=Uint8Array.from(atob(canvas.toDataURL('image/png').split(',')[1]), x=>x.charCodeAt(0)); const file=new File([bytes],'mixed scene.png',{type:'image/png'}); const dt=new DataTransfer(); dt.items.add(file); const input=document.querySelector('input[aria-label="Choose a JPG, PNG or WEBP image"]'); Object.defineProperty(input,'files',{value:dt.files}); input.dispatchEvent(new Event('change',{bubbles:true})) })()`)
await waitFor(`document.querySelector('img[alt="Preview of mixed scene.png"]')?.naturalWidth === 320`, 'image preview')
assert.equal(await evaluate(`Array.from(document.querySelectorAll('aside input[type="checkbox"]')).every(input=>input.checked)`), true, 'privacy defaults must be enabled')
assert.equal(await evaluate(`document.querySelector('aside input[value="blur"]').checked && document.querySelector('aside input[value="medium"]').checked`), true, 'blur/medium defaults must be active')

await evaluate(`Array.from(document.querySelectorAll('button')).find(button=>button.innerText.includes('Analyze Privacy')).click()`)
await waitFor(`document.body.innerText.includes('DETECTION #1') && document.body.innerText.includes('Background faces: Face 2')`, 'analysis review')
assert.equal(await evaluate(`document.body.innerText.includes('Privacy Risk Score') && document.body.innerText.includes('/ 100') && document.body.innerText.includes('CRITICAL RISK') && document.body.innerText.includes('Identity Documents')`), true, 'real document-aware risk dashboard must be visible')
assert.equal(await evaluate(`Array.from(document.querySelectorAll('label')).some(label=>label.innerText.includes('Show Identity Documents'))`), true, 'document overlay toggle must be visible')
assert.equal(await evaluate(`Array.from(document.querySelectorAll('label')).some(label=>label.innerText.includes('Show QR Codes')) && Array.from(document.querySelectorAll('label')).some(label=>label.innerText.includes('Show Barcodes'))`), true, 'QR and barcode overlay toggles must be visible')
assert.equal(await evaluate(`document.body.innerText.includes('Payment information encoded') && document.body.innerText.includes('********3457') && !document.body.innerText.includes('upi://')`), true, 'only masked code content may be displayed')
await evaluate(`Array.from(document.querySelectorAll('button')).find(button=>button.innerText.trim()==='Protect Image').click()`)
await waitFor(`document.body.innerText.toLowerCase().includes('protection complete')`, 'protected result')
assert.equal(await evaluate(`document.body.innerText.includes('7 privacy regions protected') && document.body.innerText.includes('Main subject preserved')`), true)
assert.equal(await evaluate(`document.body.innerText.includes('BEFORE PROTECTION') && document.body.innerText.includes('AFTER PROTECTION') && document.body.innerText.includes('88 points') && document.body.innerText.includes('92.6% reduction')`), true, 'document-aware risk reduction must be displayed')
assert.equal(await evaluate(`document.querySelector('img[alt^="Original"]').naturalWidth===320 && document.querySelector('img[alt^="Privacy-protected"]').naturalWidth===320`), true)
assert.equal(await evaluate(`document.querySelector('img[alt^="Original"]').getBoundingClientRect().left < document.querySelector('img[alt^="Privacy-protected"]').getBoundingClientRect().left`), true, 'desktop comparison must be side by side')
const desktopClip = await evaluate(`(() => { const r=document.querySelector('section[aria-labelledby="comparison-title"]').getBoundingClientRect(); return {x:r.left,y:r.top+window.scrollY,width:r.width,height:r.height,scale:1} })()`)
const desktopScreenshot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip: desktopClip })
if (process.env.UPDATE_SCREENSHOTS === '1') await writeFile('../docs/day8-desktop-protection.png', Buffer.from(desktopScreenshot.data, 'base64'))

await evaluate(`Array.from(document.querySelectorAll('aside label')).find(label=>label.innerText.trim()==='Pixelate').click(); Array.from(document.querySelectorAll('aside label')).find(label=>label.innerText.trim()==='High').click()`)
assert.equal(await evaluate(`!document.body.innerText.toLowerCase().includes('protection complete')`), true, 'changed settings must clear stale output')
await evaluate(`Array.from(document.querySelectorAll('button')).find(button=>button.innerText.trim()==='Protect Image').click()`)
await waitFor(`document.body.innerText.includes('pixelate · high')`, 'pixelate high result')

await evaluate(`Array.from(document.querySelectorAll('aside label')).find(label=>label.innerText.includes('Protect License Plates')).click()`)
await evaluate(`Array.from(document.querySelectorAll('button')).find(button=>button.innerText.trim()==='Protect Image').click()`)
await waitFor(`document.body.innerText.includes('6 privacy regions protected')`, 'toggle-specific result')
assert.equal(await evaluate(`Array.from(document.querySelectorAll('section[aria-labelledby="comparison-title"] div')).some(div=>div.innerText.trim()==='0 license plates')`), true)

await send('Emulation.setDeviceMetricsOverride', { width: 375, height: 844, deviceScaleFactor: 1, mobile: true })
assert.equal(await evaluate(`document.body.scrollWidth <= window.innerWidth`), true)
assert.equal(await evaluate(`document.querySelector('img[alt^="Original"]').getBoundingClientRect().top < document.querySelector('img[alt^="Privacy-protected"]').getBoundingClientRect().top`), true, 'mobile comparison must stack')
const mobileClip = await evaluate(`(() => { const r=document.querySelector('section[aria-labelledby="comparison-title"]').getBoundingClientRect(); return {x:r.left,y:r.top+window.scrollY,width:r.width,height:r.height,scale:1} })()`)
const mobileScreenshot = await send('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip: mobileClip })
if (process.env.UPDATE_SCREENSHOTS === '1') await writeFile('../docs/day8-mobile-protection.png', Buffer.from(mobileScreenshot.data, 'base64'))

await evaluate(`window.__downloadName=''; HTMLAnchorElement.prototype.click=function(){window.__downloadName=this.download}; Array.from(document.querySelectorAll('button')).find(button=>button.innerText.includes('Download Protected Image')).click()`)
assert.equal(await evaluate(`window.__downloadName`), 'privacy-protected-mixed-scene.png')
assert.deepEqual(errors, [], `Browser errors: ${errors.join('; ')}`)
socket.close()
console.log('Day 11 browser flow passed: QR/barcode overlays, masked content, protection, download, and 375/768/1024/1440 responsive layouts.')
