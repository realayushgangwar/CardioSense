import { ShieldAlert } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-[#1E3350] bg-[#060E1A]">
      <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-3 px-4 py-5 text-xs text-slate-400 sm:flex-row sm:items-center sm:px-6 lg:px-8">
        <div className="flex items-center gap-2 text-slate-300">
          <ShieldAlert className="h-4 w-4 shrink-0 text-teal-400" />
          <span className="font-medium">
            Decision-support prototype, not medical advice.
          </span>
        </div>
        <p className="text-slate-500">
          CardioSense · Framingham Heart Study ML Pipeline &amp; SHAP Explainability
        </p>
      </div>
    </footer>
  )
}

