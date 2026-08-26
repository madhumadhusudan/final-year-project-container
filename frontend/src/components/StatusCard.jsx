const stateStyles = {
  online: 'bg-emerald-400 shadow-emerald-400/60',
  connected: 'bg-emerald-400 shadow-emerald-400/60',
  checking: 'animate-pulse bg-amber-300 shadow-amber-300/60',
  offline: 'bg-rose-400 shadow-rose-400/60',
}

function StatusCard({ label, status, state }) {
  return (
    <article className="flex min-w-0 items-center justify-between gap-4 rounded-2xl border border-white/10 bg-slate-900/60 p-5">
      <div className="min-w-0">
        <p className="text-sm text-slate-400">{label}</p>
        <p className="mt-1 truncate text-lg font-semibold text-slate-100" aria-live="polite">
          {status}
        </p>
      </div>
      <span
        className={`h-3 w-3 shrink-0 rounded-full shadow-[0_0_14px_2px] ${stateStyles[state]}`}
        aria-hidden="true"
      />
    </article>
  )
}

export default StatusCard

