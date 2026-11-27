import { useEffect, useState } from 'react'
import { AlertCircle, BarChart3, Loader2, ShieldAlert, TrendingUp } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { fetchMetrics } from '../api/client'

function ConfusionHeatmap({ cm }) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp || 1
  const maxVal = Math.max(cm.tn, cm.fp, cm.fn, cm.tp, 1)

  const cellBg = (val, rgb) => {
    const alpha = (0.14 + 0.56 * (val / maxVal)).toFixed(2)
    return `rgba(${rgb}, ${alpha})`
  }

  const cells = [
    {
      label: 'True Negative (TN)',
      sub: 'Actual 0 · Pred 0',
      value: cm.tn,
      rgb: '20, 184, 166', // teal
    },
    {
      label: 'False Positive (FP)',
      sub: 'Actual 0 · Pred 1',
      value: cm.fp,
      rgb: '245, 158, 11', // amber
    },
    {
      label: 'False Negative (FN)',
      sub: 'Actual 1 · Pred 0',
      value: cm.fn,
      rgb: '239, 68, 68', // red
    },
    {
      label: 'True Positive (TP)',
      sub: 'Actual 1 · Pred 1',
      value: cm.tp,
      rgb: '16, 185, 129', // green
    },
  ]

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3 text-center">
        {cells.map((c) => (
          <div
            key={c.label}
            style={{ backgroundColor: cellBg(c.value, c.rgb) }}
            className="rounded-xl border border-[#243B5A] p-4 transition-transform duration-150 hover:scale-[1.01]"
          >
            <p className="text-xs font-semibold text-slate-200">{c.label}</p>
            <p className="mt-1 text-3xl font-extrabold text-white">{c.value}</p>
            <p className="mt-1 text-[11px] text-slate-200/80">
              {((c.value / total) * 100).toFixed(1)}% · {c.sub}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ModelPerformance() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetchMetrics()
      .then((res) => {
        if (active) setData(res)
      })
      .catch((err) => {
        if (active) setError(err.message || 'Could not load model metrics from /api/metrics.')
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  if (loading) {
    return (
      <div className="mx-auto flex max-w-7xl items-center justify-center px-4 py-24 text-slate-400">
        <Loader2 className="mr-2 h-5 w-5 animate-spin text-teal-400" />
        Loading evaluation metrics…
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="flex items-start gap-3 rounded-2xl border border-red-500/40 bg-red-500/10 p-5 text-sm text-red-200">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
          <div>
            <p className="font-semibold text-red-300">Unable to Load Model Metrics</p>
            <p className="mt-1">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  const cm = data.confusion_matrix
  const tm = data.test_metrics || {}
  const roc = data.roc_curve
  const chartRows = (data.model_comparison || []).map((row) => ({
    Model: row.Model,
    'CV ROC-AUC': row['CV ROC-AUC'],
    'Test ROC-AUC': row['Test ROC-AUC'],
    Recall: row.Recall,
    'F1-Score': row['F1-Score'],
  }))

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          Model Performance &amp; Threshold Analysis
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Stratified 80/20 split (3,390 train / 848 test) with 5-fold CV and SMOTE confined to training folds.
        </p>
      </div>

      {/* KPI Strip */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-5">
          <p className="text-xs font-medium text-slate-400">Selected Model</p>
          <p className="mt-1.5 text-xl font-bold text-white">{data.best_model}</p>
          <p className="mt-1 text-xs text-teal-400">Highest Test ROC-AUC</p>
        </div>
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-5">
          <p className="text-xs font-medium text-slate-400">Test ROC-AUC</p>
          <p className="mt-1.5 text-xl font-bold text-white">{(tm.roc_auc ?? 0).toFixed(4)}</p>
          <p className="mt-1 text-xs text-slate-400">PR-AUC: {(tm.pr_auc ?? 0).toFixed(4)}</p>
        </div>
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-5">
          <p className="text-xs font-medium text-slate-400">Chosen Threshold</p>
          <p className="mt-1.5 text-xl font-bold text-teal-400">
            {data.chosen_threshold.toFixed(2)}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Lowered from default {data.default_threshold.toFixed(2)}
          </p>
        </div>
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-5">
          <p className="text-xs font-medium text-slate-400">Tuned CHD Recall</p>
          <p className="mt-1.5 text-xl font-bold text-white">
            {((tm.recall ?? 0) * 100).toFixed(2)}%
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Precision: {((tm.precision ?? 0) * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Why Recall Is Prioritized */}
      <div className="flex items-start gap-3.5 rounded-2xl border border-teal-500/30 bg-teal-500/10 p-5">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-teal-400" />
        <div className="text-sm leading-relaxed text-slate-200">
          <p className="font-semibold text-teal-300">
            Why Recall Is Prioritized (Chosen Threshold t = {data.chosen_threshold.toFixed(2)})
          </p>
          <p className="mt-1 text-slate-300">
            In preventive CHD screening, a <strong>False Negative</strong> (missing an at-risk patient) carries a
            much higher clinical cost than a <strong>False Positive</strong> (recommending non-invasive lifestyle
            counseling or a follow-up check). Lowering the decision cutoff from{' '}
            {data.default_threshold.toFixed(2)} to <strong>{data.chosen_threshold.toFixed(2)}</strong> on
            training out-of-fold predictions increases held-out CHD recall from{' '}
            <strong>60.47%</strong> (78/129 cases) to <strong>74.42%</strong> (96/129 cases), cutting missed
            cases from 51 down to 33 at a modest precision tradeoff (24.76% → 21.52%).
          </p>
        </div>
      </div>

      {/* Metric Bar Chart + ROC Curve */}
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 lg:col-span-6">
          <div className="mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-teal-400" />
            <h2 className="text-lg font-semibold text-white">Model Metric Bar Chart</h2>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartRows} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="#1E3350" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="Model" stroke="#CBD5E1" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#091525',
                    borderColor: '#1E3350',
                    borderRadius: '12px',
                    color: '#F1F5F9',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="CV ROC-AUC" fill="#14B8A6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Test ROC-AUC" fill="#38BDF8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" fill="#818CF8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1-Score" fill="#64748B" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 lg:col-span-6">
          <div className="mb-4 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-teal-400" />
              <h2 className="text-lg font-semibold text-white">Held-Out Test ROC Curve</h2>
            </div>
            {roc && (
              <span className="rounded-full border border-teal-500/30 bg-teal-500/10 px-2.5 py-0.5 text-xs font-semibold text-teal-300">
                AUC = {roc.auc.toFixed(4)}
              </span>
            )}
          </div>
          {roc?.points?.length ? (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={roc.points} margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
                  <CartesianGrid stroke="#1E3350" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="fpr"
                    domain={[0, 1]}
                    stroke="#94A3B8"
                    tick={{ fontSize: 11 }}
                    label={{
                      value: 'False Positive Rate',
                      position: 'insideBottom',
                      offset: -8,
                      fill: '#94A3B8',
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="tpr"
                    domain={[0, 1]}
                    stroke="#94A3B8"
                    tick={{ fontSize: 11 }}
                    label={{
                      value: 'True Positive Rate',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#94A3B8',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#091525',
                      borderColor: '#1E3350',
                      borderRadius: '12px',
                      color: '#F1F5F9',
                      fontSize: '12px',
                    }}
                    formatter={(val) => Number(val).toFixed(3)}
                    labelFormatter={(fpr) => `FPR: ${Number(fpr).toFixed(3)}`}
                  />
                  <ReferenceLine
                    segment={[
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ]}
                    stroke="#475569"
                    strokeDasharray="4 4"
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    name={roc.model}
                    stroke="#14B8A6"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="py-16 text-center text-sm text-slate-400">
              ROC curve points unavailable from API.
            </p>
          )}
        </div>
      </div>

      {/* Confusion Matrix Heatmap + Comparison Table */}
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 lg:col-span-5">
          <h2 className="text-lg font-semibold text-white">
            Confusion Matrix Heatmap (t = {cm.threshold.toFixed(2)})
          </h2>
          <p className="mb-4 mt-1 text-xs text-slate-400">
            Color intensity scales with patient count across the 848 held-out test records.
          </p>
          <ConfusionHeatmap cm={cm} />
        </div>

        <div className="overflow-hidden rounded-2xl border border-[#1E3350] bg-[#0D1B2E] lg:col-span-7">
          <div className="border-b border-[#1E3350] px-6 py-4">
            <h2 className="text-base font-semibold text-white">
              Default Threshold (t = 0.50) Held-Out Test Comparison
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-[#091525] text-xs uppercase text-slate-400">
                <tr>
                  <th className="px-5 py-3.5">Model</th>
                  <th className="px-4 py-3.5">CV AUC</th>
                  <th className="px-4 py-3.5">Test AUC</th>
                  <th className="px-4 py-3.5">PR-AUC</th>
                  <th className="px-4 py-3.5">Recall</th>
                  <th className="px-4 py-3.5">Precision</th>
                  <th className="px-4 py-3.5">F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E3350] text-slate-200">
                {data.model_comparison.map((row) => (
                  <tr key={row.Model} className="hover:bg-[#12233B]/60">
                    <td className="px-5 py-3.5 font-semibold text-white">
                      {row.Model}
                      {row.Model === data.best_model && (
                        <span className="ml-2 rounded-full bg-teal-500/15 px-2 py-0.5 text-[11px] font-medium text-teal-300">
                          Selected
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3.5">{Number(row['CV ROC-AUC']).toFixed(4)}</td>
                    <td className="px-4 py-3.5">{Number(row['Test ROC-AUC']).toFixed(4)}</td>
                    <td className="px-4 py-3.5">{Number(row['Test PR-AUC']).toFixed(4)}</td>
                    <td className="px-4 py-3.5">{Number(row.Recall).toFixed(4)}</td>
                    <td className="px-4 py-3.5">{Number(row.Precision).toFixed(4)}</td>
                    <td className="px-4 py-3.5">{Number(row['F1-Score']).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
