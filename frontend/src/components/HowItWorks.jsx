import { CheckIcon, ShieldIcon, UploadIcon } from './Icons.jsx'

const steps = [
  { title: 'Choose an image', text: 'Drop a JPG, PNG or WEBP file into the local privacy workspace.', icon: UploadIcon },
  { title: 'Review before sharing', text: 'Confirm the preview and image details before requesting analysis.', icon: CheckIcon },
  { title: 'Review real detections', text: 'Inspect face and person boxes, confidence, and explainable subject classification.', icon: ShieldIcon },
]

function HowItWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-24 border-y border-slate-200 bg-white px-4 py-16 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="max-w-2xl"><p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700">How it works</p><h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">A clear check before every share</h2></div>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {steps.map(({ title, text, icon: StepIcon }, index) => <article key={title} className="rounded-2xl border border-slate-200 p-5"><div className="flex items-center justify-between"><span className="grid h-10 w-10 place-items-center rounded-xl bg-teal-50 text-teal-800"><StepIcon className="h-5 w-5" /></span><span className="text-xs font-bold text-slate-400">0{index + 1}</span></div><h3 className="mt-5 font-bold text-slate-900">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-600">{text}</p></article>)}
        </div>
      </div>
    </section>
  )
}

export default HowItWorks
