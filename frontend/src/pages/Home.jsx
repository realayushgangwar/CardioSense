import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  ClipboardList,
  HeartPulse,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { checkHealth } from '../api/client'

const HOW_IT_WORKS = [
  {
    step: '01',
    icon: ClipboardList,
    title: 'Enter Clinical Profile',
    description:
      'Input 15 Framingham demographics, lifestyle habits, and clinical measurements with instant inline validation.',
  },
  {
    step: '02',
    icon: BrainCircuit,
    title: 'Leakage-Free Risk & SHAP',
    description:
      'Our SMOTE-trained Logistic Regression pipeline predicts 10-year CHD probability and computes exact SHAP drivers.',
  },
  {
    step: '03',
    icon: Sparkles,
    title: 'Personalized Prevention',
    description:
      'Receive severity-ranked preventive recommendations aligned with ACC/AHA and ADA guidelines for elevated factors.',
  },
]

const STATS = [
  { label: 'Framingham Cohort', value: '4,238', sub: 'Patients · 15 predictors' },
  { label: 'Tuned CHD Recall', value: '74.4%', sub: 'At screening cutoff t = 0.41' },
  { label: 'Test ROC-AUC', value: '0.6985', sub: 'Logistic Regression pipeline' },
  { label: 'Risk Tiers', value: '3 Tiers', sub: 'Low <10% · Mod 10–25% · High >25%' },
]

export default function Home() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let active = true
    checkHealth()
      .then((res) => {
        if (active) setHealth(res)
      })
      .catch(() => {
        if (active) setHealth({ status: 'offline' })
      })
    return () => {
      active = false
    }
  }, [])

  return (
    <div className="mx-auto max-w-7xl space-y-12 px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl border border-[#1E3350] bg-gradient-to-br from-[#0D1B2E] via-[#0B1728] to-[#081322] p-7 shadow-2xl sm:p-10 lg:p-12">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-teal-500/10 blur-3xl" />

        <div className="relative flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-teal-500/30 bg-teal-500/10 px-3.5 py-1 text-xs font-medium text-teal-300">
              <Activity className="h-3.5 w-3.5" />
              <span>Explainable Cardiovascular Screening Prototype</span>
              {health?.status === 'ok' && (
                <span className="ml-1 inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2 py-0.5 text-[11px] text-emerald-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  API Online
                </span>
              )}
              {health?.status === 'offline' && (
                <span className="ml-1 inline-flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-0.5 text-[11px] text-amber-300">
                  API Offline
                </span>
              )}
            </div>

            <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl lg:text-5xl">
              Early 10-Year Coronary Heart Disease Risk,{' '}
              <span className="text-teal-400">Explained Clearly.</span>
            </h1>

            <p className="text-base leading-relaxed text-slate-300 sm:text-lg">
              CardioSense combines a leakage-free Framingham machine learning pipeline with exact
              SHAP feature attributions and clinical-guideline preventive recommendations.
            </p>

            <div className="flex flex-wrap items-center gap-3.5 pt-1">
              <Link
                to="/assess"
                className="inline-flex items-center gap-2 rounded-xl bg-teal-500 px-6 py-3.5 text-sm font-semibold text-slate-950 shadow-lg shadow-teal-500/20 transition-all duration-150 hover:bg-teal-400"
              >
                <HeartPulse className="h-4 w-4" />
                Start Risk Assessment
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/performance"
                className="inline-flex items-center gap-2 rounded-xl border border-[#243B5A] bg-[#12233B] px-5 py-3.5 text-sm font-semibold text-slate-200 transition-colors duration-150 hover:border-teal-500/40 hover:text-white"
              >
                <BarChart3 className="h-4 w-4 text-teal-400" />
                View Model Performance
              </Link>
            </div>
          </div>

          <div className="grid shrink-0 grid-cols-2 gap-3.5 lg:w-[380px]">
            {STATS.map((stat) => (
              <div
                key={stat.label}
                className="rounded-2xl border border-[#1E3350] bg-[#091525]/90 p-4 transition-colors hover:border-teal-500/30"
              >
                <p className="text-xs font-medium text-slate-400">{stat.label}</p>
                <p className="mt-1 text-2xl font-bold text-white">{stat.value}</p>
                <p className="mt-1 text-[11px] text-teal-300/80">{stat.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Three-Step "How It Works" Strip */}
      <section className="space-y-5">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-teal-400">
              Workflow
            </p>
            <h2 className="text-2xl font-bold text-white">How CardioSense Works</h2>
          </div>
          <Link
            to="/assess"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-teal-400 hover:text-teal-300"
          >
            Try the assessment <ArrowRight className="h-4 w-4" />
          </Link>
        </div>

        <div className="grid gap-5 md:grid-cols-3">
          {HOW_IT_WORKS.map((item) => {
            const Icon = item.icon
            return (
              <div
                key={item.step}
                className="relative rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 transition-all duration-200 hover:-translate-y-0.5 hover:border-teal-500/40"
              >
                <div className="flex items-center justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-teal-500/30 bg-teal-500/10 text-teal-400">
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="text-2xl font-extrabold text-slate-700">
                    {item.step}
                  </span>
                </div>
                <h3 className="mt-4 text-lg font-semibold text-white">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  {item.description}
                </p>
              </div>
            )
          })}
        </div>
      </section>

      {/* Bottom Trust & CTA Strip */}
      <section className="flex flex-col items-start justify-between gap-4 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 sm:flex-row sm:items-center">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-6 w-6 shrink-0 text-teal-400" />
          <div>
            <h3 className="text-base font-semibold text-white">
              Ready to run a 10-year CHD risk check?
            </h3>
            <p className="mt-0.5 text-xs text-slate-400">
              Includes pre-loaded Low, Moderate, and High sample profiles plus printable PDF reports.
            </p>
          </div>
        </div>
        <Link
          to="/assess"
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-teal-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-400"
        >
          Go to Assess <ArrowRight className="h-4 w-4" />
        </Link>
      </section>
    </div>
  )
}
