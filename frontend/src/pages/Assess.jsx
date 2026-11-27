import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cigarette,
  Droplets,
  Flame,
  HeartPulse,
  HelpCircle,
  Loader2,
  Printer,
  Scale,
  Stethoscope,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchFeatures, predictRisk } from '../api/client'

const SECTION_ORDER = ['Demographics', 'Lifestyle', 'Clinical Measurements']

const SAMPLE_PRESETS = {
  low: {
    label: 'Low',
    data: {
      age: 28,
      male: 0,
      education: 3,
      currentSmoker: 0,
      cigsPerDay: 0,
      sysBP: 112,
      diaBP: 72,
      totChol: 175,
      BMI: 22.0,
      heartRate: 68,
      glucose: 82,
      prevalentHyp: 0,
      BPMeds: 0,
      diabetes: 0,
      prevalentStroke: 0,
    },
  },
  moderate: {
    label: 'Moderate',
    data: {
      age: 40,
      male: 0,
      education: 2,
      currentSmoker: 0,
      cigsPerDay: 0,
      sysBP: 132,
      diaBP: 84,
      totChol: 220,
      BMI: 26.2,
      heartRate: 74,
      glucose: 94,
      prevalentHyp: 0,
      BPMeds: 0,
      diabetes: 0,
      prevalentStroke: 0,
    },
  },
  high: {
    label: 'High',
    data: {
      age: 65,
      male: 1,
      education: 1,
      currentSmoker: 1,
      cigsPerDay: 30,
      sysBP: 165,
      diaBP: 98,
      totChol: 270,
      BMI: 30.5,
      heartRate: 80,
      glucose: 180,
      prevalentHyp: 1,
      BPMeds: 1,
      diabetes: 1,
      prevalentStroke: 0,
    },
  },
}

const RISK_THEME = {
  Low: {
    badge: 'LOW RISK',
    text: 'text-emerald-400',
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-500/10',
    hex: '#10B981',
    summary:
      'indicates a low estimated 10-year likelihood of developing coronary heart disease. Maintaining current healthy habits and routine preventive checkups is recommended.',
  },
  Moderate: {
    badge: 'MODERATE RISK',
    text: 'text-amber-400',
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/10',
    hex: '#F59E0B',
    summary:
      'places this profile in the moderate 10-year coronary heart disease risk tier. Addressing any elevated lifestyle or metabolic factors early can meaningfully lower long-term risk.',
  },
  High: {
    badge: 'HIGH RISK',
    text: 'text-red-400',
    border: 'border-red-500/40',
    bg: 'bg-red-500/10',
    hex: '#EF4444',
    summary:
      'places this profile in the high 10-year coronary heart disease risk tier. A proactive clinical consultation to review blood pressure, lipids, glucose, or smoking cessation is strongly advised.',
  },
}

const SEVERITY_ORDER = { High: 3, Elevated: 2, Borderline: 1 }

function getCategoryIcon(category) {
  if (category.includes('Smoking')) return Cigarette
  if (category.includes('Blood Pressure')) return HeartPulse
  if (category.includes('Lipid')) return Droplets
  if (category.includes('Weight') || category.includes('BMI')) return Scale
  return Flame
}

function AnimatedGauge({ probability, riskLevel }) {
  const theme = RISK_THEME[riskLevel] || RISK_THEME.Moderate
  const [animatedPct, setAnimatedPct] = useState(0)
  const targetPct = Math.max(0, Math.min(100, probability * 100))

  useEffect(() => {
    const id = requestAnimationFrame(() => setAnimatedPct(targetPct))
    return () => cancelAnimationFrame(id)
  }, [targetPct])

  const radius = 84
  const circumference = Math.PI * radius
  const dashOffset = circumference * (1 - animatedPct / 100)

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 220 130" className="w-full max-w-[230px]">
        {/* Background track */}
        <path
          d="M 26 110 A 84 84 0 0 1 194 110"
          fill="none"
          stroke="#1E3350"
          strokeWidth="18"
          strokeLinecap="round"
        />
        {/* Animated foreground arc */}
        <path
          d="M 26 110 A 84 84 0 0 1 194 110"
          fill="none"
          stroke={theme.hex}
          strokeWidth="18"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          style={{ transition: 'stroke-dashoffset 700ms cubic-bezier(0.22, 1, 0.36, 1)' }}
        />
        <text
          x="110"
          y="88"
          textAnchor="middle"
          fill={theme.hex}
          fontSize="28"
          fontWeight="800"
        >
          {targetPct.toFixed(1)}%
        </text>
        <text
          x="110"
          y="110"
          textAnchor="middle"
          fill="#94A3B8"
          fontSize="11"
          fontWeight="600"
        >
          {riskLevel} Risk Tier
        </text>
      </svg>
    </div>
  )
}

function InfoTooltip({ text }) {
  return (
    <span className="group relative inline-flex items-center">
      <HelpCircle className="h-3.5 w-3.5 cursor-help text-slate-500 transition-colors group-hover:text-teal-400" />
      <span className="pointer-events-none absolute right-0 top-5 z-30 hidden w-56 rounded-lg border border-[#243B5A] bg-[#07111F] p-2.5 text-[11px] leading-relaxed text-slate-200 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  )
}

export default function Assess() {
  const [features, setFeatures] = useState([])
  const [formValues, setFormValues] = useState({})
  const [loadingFeatures, setLoadingFeatures] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [apiError, setApiError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    let active = true
    fetchFeatures()
      .then((res) => {
        if (!active) return
        const list = res.features || []
        setFeatures(list)
        const defaults = {}
        list.forEach((f) => {
          defaults[f.name] = f.default
        })
        setFormValues(defaults)
      })
      .catch((err) => {
        if (active) setApiError(err.message || 'Could not load feature schema from /api/features.')
      })
      .finally(() => {
        if (active) setLoadingFeatures(false)
      })
    return () => {
      active = false
    }
  }, [])

  const groupedFeatures = useMemo(() => {
    const groups = {}
    SECTION_ORDER.forEach((s) => {
      groups[s] = []
    })
    features.forEach((f) => {
      const sec = f.section || 'Clinical Measurements'
      if (!groups[sec]) groups[sec] = []
      groups[sec].push(f)
    })
    return groups
  }, [features])

  // Inline validation computed live from current formValues
  const fieldErrors = useMemo(() => {
    const errs = {}
    features.forEach((f) => {
      if (f.type === 'number' || f.type === 'integer') {
        const num = Number(formValues[f.name])
        if (formValues[f.name] === '' || Number.isNaN(num)) {
          errs[f.name] = 'Enter a valid number.'
        } else if (num < f.min || num > f.max) {
          errs[f.name] = `Must be between ${f.min} and ${f.max}.`
        }
      }
    })

    const sys = Number(formValues.sysBP)
    const dia = Number(formValues.diaBP)
    if (!Number.isNaN(sys) && !Number.isNaN(dia) && dia >= sys) {
      errs.diaBP = `Diastolic BP (${dia}) must be lower than Systolic BP (${sys}).`
    }

    if (Number(formValues.currentSmoker) === 0 && Number(formValues.cigsPerDay) > 0) {
      errs.cigsPerDay = 'Set to 0 for non-smokers or enable Current Smoker.'
    }
    return errs
  }, [features, formValues])

  const hasValidationErrors = Object.keys(fieldErrors).length > 0

  const handleFieldChange = (name, value) => {
    setFormValues((prev) => ({ ...prev, [name]: value }))
  }

  const applyPreset = (key) => {
    setApiError('')
    setFormValues((prev) => ({ ...prev, ...SAMPLE_PRESETS[key].data }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setApiError('')
    if (hasValidationErrors) return

    setSubmitting(true)
    try {
      const payload = {}
      features.forEach((f) => {
        const raw = formValues[f.name]
        payload[f.name] =
          f.type === 'integer' ||
          f.type === 'boolean' ||
          (f.type === 'select' && f.name === 'male')
            ? parseInt(raw, 10)
            : parseFloat(raw)
      })
      const res = await predictRisk(payload)
      setResult(res)
    } catch (err) {
      setApiError(err.message || 'Prediction request failed.')
    } finally {
      setSubmitting(false)
    }
  }

  const riskTheme = result ? RISK_THEME[result.risk_level] || RISK_THEME.Moderate : null

  const shapChartData = useMemo(() => {
    if (!result?.shap_contributions) return []
    return [...result.shap_contributions]
      .reverse()
      .map((item) => ({
        ...item,
        shortLabel: item.label.replace(/\s*\([^)]*\)/g, ''),
      }))
  }, [result])

  const sortedRecommendations = useMemo(() => {
    if (!result?.recommendations) return []
    return [...result.recommendations].sort(
      (a, b) => (SEVERITY_ORDER[b.severity] || 0) - (SEVERITY_ORDER[a.severity] || 0)
    )
  }, [result])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Page Header + Sample Presets */}
      <div className="no-print mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
            10-Year CHD Risk Assessment
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Complete the patient profile below to compute predicted risk, SHAP drivers, and tailored guidance.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-slate-400">Load sample:</span>
          <button
            type="button"
            onClick={() => applyPreset('low')}
            className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-300 transition hover:bg-emerald-500/20"
          >
            Low
          </button>
          <button
            type="button"
            onClick={() => applyPreset('moderate')}
            className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-300 transition hover:bg-amber-500/20"
          >
            Moderate
          </button>
          <button
            type="button"
            onClick={() => applyPreset('high')}
            className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs font-semibold text-red-300 transition hover:bg-red-500/20"
          >
            High
          </button>
        </div>
      </div>

      {apiError && (
        <div className="no-print mb-6 flex items-start gap-3 rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-200">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
          <div>
            <p className="font-semibold text-red-300">Unable to Complete Request</p>
            <p className="mt-0.5">{apiError}</p>
          </div>
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-12">
        {/* Left: Dynamic Form */}
        <form
          onSubmit={handleSubmit}
          className="no-print space-y-6 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-5 sm:p-6 lg:col-span-5"
        >
          {loadingFeatures ? (
            <div className="flex items-center justify-center py-20 text-slate-400">
              <Loader2 className="mr-2 h-5 w-5 animate-spin text-teal-400" />
              Loading form fields from API…
            </div>
          ) : (
            <>
              {SECTION_ORDER.map((section) => {
                const list = groupedFeatures[section] || []
                if (!list.length) return null
                return (
                  <div key={section} className="space-y-4">
                    <h2 className="border-b border-[#1E3350] pb-2 text-xs font-bold uppercase tracking-wider text-teal-400">
                      {section}
                    </h2>
                    <div className="space-y-3.5">
                      {list.map((f) => {
                        const val = formValues[f.name] ?? f.default
                        const fieldErr = fieldErrors[f.name]

                        // Toggle for boolean fields
                        if (f.type === 'boolean') {
                          const enabled = Number(val) === 1
                          return (
                            <div
                              key={f.name}
                              className="flex items-center justify-between rounded-xl border border-[#1E3350] bg-[#091525] px-3.5 py-2.5"
                            >
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-medium text-slate-200">
                                  {f.label}
                                </span>
                                <InfoTooltip text={f.description} />
                              </div>
                              <button
                                type="button"
                                role="switch"
                                aria-checked={enabled}
                                onClick={() => handleFieldChange(f.name, enabled ? 0 : 1)}
                                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                                  enabled ? 'bg-teal-500' : 'bg-slate-700'
                                }`}
                              >
                                <span
                                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition duration-200 ${
                                    enabled ? 'translate-x-5' : 'translate-x-0'
                                  }`}
                                />
                              </button>
                            </div>
                          )
                        }

                        // Select dropdown
                        if (f.type === 'select' && f.options) {
                          return (
                            <div key={f.name} className="space-y-1.5">
                              <div className="flex items-center justify-between">
                                <label className="text-xs font-medium text-slate-300">
                                  {f.label}
                                </label>
                                <InfoTooltip text={f.description} />
                              </div>
                              <select
                                value={val}
                                onChange={(e) => handleFieldChange(f.name, e.target.value)}
                                className="w-full rounded-xl border border-[#1E3350] bg-[#091525] px-3 py-2 text-sm text-slate-100 focus:border-teal-400 focus:outline-none"
                              >
                                {f.options.map((opt) => (
                                  <option key={opt.value} value={opt.value}>
                                    {opt.label}
                                  </option>
                                ))}
                              </select>
                            </div>
                          )
                        }

                        // Slider + Number Input for numeric/integer fields
                        return (
                          <div key={f.name} className="space-y-1.5">
                            <div className="flex items-center justify-between gap-2">
                              <label className="text-xs font-medium text-slate-300">
                                {f.label}{' '}
                                <span className="text-[11px] text-slate-500">({f.unit})</span>
                              </label>
                              <InfoTooltip text={f.description} />
                            </div>
                            <div className="flex items-center gap-3">
                              <input
                                type="range"
                                min={f.min}
                                max={f.max}
                                step={f.step}
                                value={Number.isNaN(Number(val)) ? f.min : val}
                                onChange={(e) => handleFieldChange(f.name, e.target.value)}
                                className="h-1.5 flex-1 cursor-pointer appearance-none rounded-lg bg-[#1E3350] accent-teal-400"
                              />
                              <input
                                type="number"
                                min={f.min}
                                max={f.max}
                                step={f.step}
                                value={val}
                                onChange={(e) => handleFieldChange(f.name, e.target.value)}
                                className={`w-24 rounded-xl border bg-[#091525] px-2.5 py-1.5 text-right text-sm text-slate-100 focus:outline-none ${
                                  fieldErr
                                    ? 'border-red-500/70 focus:border-red-400'
                                    : 'border-[#1E3350] focus:border-teal-400'
                                }`}
                              />
                            </div>
                            {fieldErr && (
                              <p className="text-[11px] font-medium text-red-400">{fieldErr}</p>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })}

              <button
                type="submit"
                disabled={submitting || hasValidationErrors}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-teal-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-teal-500/20 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing Clinical Profile…
                  </>
                ) : (
                  <>
                    <Activity className="h-4 w-4" />
                    Predict 10-Year CHD Risk
                  </>
                )}
              </button>
            </>
          )}
        </form>

        {/* Right: Results */}
        <div className="print-area space-y-6 lg:col-span-7">
          {submitting ? (
            <div className="flex min-h-[420px] flex-col items-center justify-center rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-8 text-center">
              <Loader2 className="h-10 w-10 animate-spin text-teal-400" />
              <p className="mt-4 text-base font-semibold text-white">
                Computing 10-Year Risk &amp; SHAP Explanations…
              </p>
              <p className="mt-1 text-xs text-slate-400">
                Running the leakage-free Logistic Regression pipeline and LinearExplainer.
              </p>
            </div>
          ) : !result ? (
            <div className="flex min-h-[420px] flex-col items-center justify-center rounded-2xl border border-dashed border-[#1E3350] bg-[#0D1B2E]/60 p-8 text-center">
              <Stethoscope className="h-12 w-12 text-teal-400/75" />
              <h2 className="mt-4 text-lg font-semibold text-white">
                No Prediction Generated Yet
              </h2>
              <p className="mt-2 max-w-md text-sm text-slate-400">
                Adjust the sliders and toggles on the left or click{' '}
                <strong className="text-teal-300">Load sample: Low / Moderate / High</strong>, then
                submit to inspect risk probability, SHAP drivers, and preventive recommendations.
              </p>
            </div>
          ) : (
            <>
              {/* Action bar with Download Report */}
              <div className="no-print flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400">
                  Assessment complete · Threshold cutoff {(result.threshold * 100).toFixed(0)}%
                </span>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="inline-flex items-center gap-2 rounded-xl border border-[#243B5A] bg-[#12233B] px-4 py-2 text-xs font-semibold text-slate-200 transition hover:border-teal-400/50 hover:text-white"
                >
                  <Printer className="h-4 w-4 text-teal-400" />
                  Download report
                </button>
              </div>

              {/* 1. Large Risk Card + Animated Gauge */}
              <div
                className={`print-card rounded-2xl border ${riskTheme.border} bg-[#0D1B2E] p-6 shadow-xl`}
              >
                <div className="grid items-center gap-6 sm:grid-cols-12">
                  <div className="space-y-3 sm:col-span-7">
                    <span
                      className={`inline-block rounded-full border ${riskTheme.border} ${riskTheme.bg} px-3 py-1 text-xs font-bold tracking-wider ${riskTheme.text}`}
                    >
                      {riskTheme.badge}
                    </span>
                    <div className="text-4xl font-extrabold text-white">
                      {(result.probability * 100).toFixed(1)}%
                      <span className="ml-2 text-sm font-normal text-slate-400">
                        10-Year CHD Probability
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-slate-300">
                      An estimated 10-year probability of{' '}
                      <strong className={riskTheme.text}>
                        {(result.probability * 100).toFixed(1)}%
                      </strong>{' '}
                      {riskTheme.summary}
                    </p>
                    <p className="text-xs text-slate-400">
                      Screening threshold: {(result.threshold * 100).toFixed(0)}% · Tier cutoffs:
                      Low &lt;10% · Moderate 10–25% · High &gt;25%
                    </p>
                  </div>

                  <div className="sm:col-span-5">
                    <AnimatedGauge probability={result.probability} riskLevel={result.risk_level} />
                  </div>
                </div>
              </div>

              {/* 2. "Why this result" Diverging SHAP Bar Chart */}
              <div className="print-card rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6">
                <h2 className="text-lg font-semibold text-white">Why this result</h2>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">
                  How to read: bars extending right in <span className="text-red-400 font-semibold">red</span> push
                  predicted risk upward; bars extending left in <span className="text-emerald-400 font-semibold">green</span> pull
                  predicted risk downward relative to the Framingham training baseline.
                </p>

                <div className="mt-4 h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={shapChartData}
                      layout="vertical"
                      margin={{ top: 6, right: 20, left: 10, bottom: 6 }}
                    >
                      <CartesianGrid stroke="#1E3350" strokeDasharray="3 3" horizontal={false} />
                      <XAxis
                        type="number"
                        stroke="#94A3B8"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(v) => v.toFixed(2)}
                      />
                      <YAxis
                        type="category"
                        dataKey="shortLabel"
                        width={145}
                        stroke="#CBD5E1"
                        tick={{ fontSize: 11 }}
                      />
                      <ReferenceLine x={0} stroke="#64748B" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#091525',
                          borderColor: '#1E3350',
                          borderRadius: '12px',
                          color: '#F1F5F9',
                          fontSize: '12px',
                        }}
                        formatter={(val, _name, props) => [
                          `${Number(val) >= 0 ? '+' : ''}${Number(val).toFixed(4)} log-odds (value: ${props.payload.value ?? 'N/A'})`,
                          props.payload.direction,
                        ]}
                      />
                      <Bar dataKey="shap" radius={[4, 4, 4, 4]}>
                        {shapChartData.map((entry) => (
                          <Cell
                            key={entry.feature}
                            fill={entry.shap >= 0 ? '#EF4444' : '#10B981'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 3. Recommendations sorted by severity */}
              <div className="print-card rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6">
                <h2 className="mb-4 text-lg font-semibold text-white">Recommendations</h2>
                {sortedRecommendations.length === 0 ? (
                  <div className="flex items-start gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-200">
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-emerald-300">
                        All Modifiable Factors Within Guideline Range
                      </p>
                      <p className="mt-1 text-emerald-200/80">
                        No elevated smoking, blood pressure, cholesterol, BMI, or glucose targets detected.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-3.5 sm:grid-cols-2">
                    {sortedRecommendations.map((rec) => {
                      const Icon = getCategoryIcon(rec.category)
                      const isHigh = rec.severity === 'High'
                      return (
                        <div
                          key={rec.category}
                          className="print-card rounded-xl border border-[#1E3350] bg-[#091525] p-4 transition hover:border-teal-500/35"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/10 text-teal-400">
                                <Icon className="h-4 w-4" />
                              </div>
                              <h3 className="text-sm font-semibold text-white">{rec.title}</h3>
                            </div>
                            <span
                              className={`shrink-0 rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                                isHigh
                                  ? 'border border-red-500/30 bg-red-500/15 text-red-300'
                                  : 'border border-amber-500/30 bg-amber-500/15 text-amber-300'
                              }`}
                            >
                              {rec.severity}
                            </span>
                          </div>
                          <p className="mt-2.5 text-xs leading-relaxed text-slate-400">
                            {rec.advice}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
