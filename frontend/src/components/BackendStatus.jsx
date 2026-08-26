const statusContent = {
  checking: { label: 'Checking backend', dot: 'bg-amber-400 animate-pulse', text: 'text-amber-800', background: 'bg-amber-50 border-amber-200' },
  connected: { label: 'Backend connected', dot: 'bg-emerald-500', text: 'text-emerald-800', background: 'bg-emerald-50 border-emerald-200' },
  offline: { label: 'Backend offline', dot: 'bg-rose-500', text: 'text-rose-800', background: 'bg-rose-50 border-rose-200' },
}

function BackendStatus({ status }) {
  const content = statusContent[status] || statusContent.checking
  return <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${content.background} ${content.text}`} role="status" aria-live="polite"><span className={`h-2 w-2 rounded-full ${content.dot}`} aria-hidden="true" /><span>{content.label}</span></div>
}

export default BackendStatus
