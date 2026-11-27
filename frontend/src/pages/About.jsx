import {
  AlertTriangle,
  BrainCircuit,
  Code2,
  Database,
  GitBranch,
  Shield,
} from 'lucide-react'

const PIPELINE_STEPS = [
  '1. Stratified 80/20 Split (3,390 train / 848 test, seed 42)',
  '2. ColumnTransformer (median + StandardScaler numeric; mode categorical)',
  '3. SMOTE inside imblearn.Pipeline (training folds only)',
  '4. 5-Fold CV Tuning (Logistic Regression, Random Forest, XGBoost)',
  '5. OOF Threshold Tuning (t = 0.50 → 0.41 for 74.42% recall)',
]

export default function About() {
  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          About CardioSense
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Dataset overview, leakage-free ML pipeline, SHAP vs. LIME explainability, limitations, and project stack.
        </p>
      </div>

      {/* Dataset + Methodology Pipeline */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6">
          <div className="flex items-center gap-2.5 text-teal-400">
            <Database className="h-5 w-5" />
            <h2 className="text-lg font-semibold text-white">Dataset</h2>
          </div>
          <ul className="space-y-2.5 text-sm leading-relaxed text-slate-300">
            <li>
              <strong className="text-white">Source:</strong> Framingham Heart Study (4,238 patient records, 15 clinical &amp; demographic predictors, binary target <code className="text-teal-300">TenYearCHD</code>).
            </li>
            <li>
              <strong className="text-white">Class Split:</strong> 3,594 No CHD (84.8%) vs. 644 10-Year CHD (15.2%) — a 5.58 : 1 imbalance ratio.
            </li>
            <li>
              <strong className="text-white">Missing Data:</strong> 645 missing cells across 7 columns (led by <code className="text-teal-300">glucose</code> at 9.16% and <code className="text-teal-300">education</code> at 2.48%), imputed strictly inside the training pipeline.
            </li>
            <li>
              <strong className="text-white">Duplicates:</strong> 0 exact duplicate rows and 0 duplicate feature vectors.
            </li>
          </ul>
        </div>

        <div className="space-y-3 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6">
          <div className="flex items-center gap-2.5 text-teal-400">
            <GitBranch className="h-5 w-5" />
            <h2 className="text-lg font-semibold text-white">Methodology Pipeline</h2>
          </div>
          <div className="space-y-2">
            {PIPELINE_STEPS.map((step) => (
              <div
                key={step}
                className="rounded-xl border border-[#1E3350] bg-[#091525] px-3.5 py-2 text-xs font-medium text-slate-200"
              >
                {step}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SHAP vs LIME Comparison */}
      <div className="overflow-hidden rounded-2xl border border-[#1E3350] bg-[#0D1B2E]">
        <div className="flex items-center gap-2.5 border-b border-[#1E3350] px-6 py-4">
          <BrainCircuit className="h-5 w-5 text-teal-400" />
          <h2 className="text-lg font-semibold text-white">SHAP vs. LIME Explainability</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-[#091525] text-xs uppercase text-slate-400">
              <tr>
                <th className="px-6 py-3.5">Dimension</th>
                <th className="px-6 py-3.5">SHAP (LinearExplainer)</th>
                <th className="px-6 py-3.5">LIME (LimeTabularExplainer)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E3350] text-slate-300">
              <tr>
                <td className="px-6 py-3.5 font-semibold text-white">Theoretical Basis</td>
                <td className="px-6 py-3.5">
                  Exact Shapley values from cooperative game theory (local accuracy, consistency, missingness).
                </td>
                <td className="px-6 py-3.5">
                  Approximate local linear surrogate fitted on randomly perturbed neighbors.
                </td>
              </tr>
              <tr>
                <td className="px-6 py-3.5 font-semibold text-white">Determinism</td>
                <td className="px-6 py-3.5 text-emerald-300">100% deterministic for Logistic Regression.</td>
                <td className="px-6 py-3.5">Stochastic (varies slightly across perturbation samples).</td>
              </tr>
              <tr>
                <td className="px-6 py-3.5 font-semibold text-white">Extreme Values</td>
                <td className="px-6 py-3.5">
                  Preserves exact standardized magnitude (e.g., severe hypertension receives full weight).
                </td>
                <td className="px-6 py-3.5">
                  Discretizes continuous features into quartiles, compressing extreme readings.
                </td>
              </tr>
              <tr>
                <td className="px-6 py-3.5 font-semibold text-white">Role in CardioSense</td>
                <td className="px-6 py-3.5">
                  Primary attribution engine powering <code className="text-teal-300">/api/predict</code> and dashboards.
                </td>
                <td className="px-6 py-3.5">
                  Complementary rule-based sanity check in <code className="text-teal-300">03_explainability.ipynb</code>.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Limitations & Project Info */}
      <div className="grid gap-6 md:grid-cols-3">
        <div className="space-y-3 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6 md:col-span-2">
          <div className="flex items-center gap-2.5 text-amber-400">
            <AlertTriangle className="h-5 w-5" />
            <h2 className="text-lg font-semibold text-white">Limitations &amp; Ethics</h2>
          </div>
          <div className="grid gap-3.5 sm:grid-cols-2 text-sm text-slate-300">
            <div className="rounded-xl border border-[#1E3350] bg-[#091525] p-4">
              <p className="font-semibold text-white">Single-Population Cohort</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">
                Framingham participants were predominantly white residents of Framingham, MA; risk weights may not generalize uniformly to diverse modern populations.
              </p>
            </div>
            <div className="rounded-xl border border-[#1E3350] bg-[#091525] p-4">
              <p className="font-semibold text-white">Not a Diagnostic Tool</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">
                Academic decision-support prototype only; never a substitute for physician evaluation or clinical testing.
              </p>
            </div>
            <div className="rounded-xl border border-[#1E3350] bg-[#091525] p-4">
              <p className="font-semibold text-white">Class Imbalance Tradeoff</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">
                Prioritizing 74.42% CHD recall lowers precision to 21.52%; positive screening flags should prompt routine follow-up rather than alarm.
              </p>
            </div>
            <div className="rounded-xl border border-[#1E3350] bg-[#091525] p-4">
              <p className="flex items-center gap-1.5 font-semibold text-white">
                <Shield className="h-4 w-4 text-teal-400" />
                SMOTE Calibration Shift
              </p>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">
                Training on SMOTE-balanced folds shifts raw probabilities upward relative to the 15.2% base rate; post-hoc calibration is planned for future work.
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-3 rounded-2xl border border-[#1E3350] bg-[#0D1B2E] p-6">
          <div className="flex items-center gap-2.5 text-teal-400">
            <Code2 className="h-5 w-5" />
            <h2 className="text-lg font-semibold text-white">Project Info</h2>
          </div>
          <div className="space-y-2.5 text-xs leading-relaxed text-slate-300">
            <p>
              <strong className="text-white">Project:</strong> CardioSense (College Mini Project)
            </p>
            <p>
              <strong className="text-white">ML &amp; XAI:</strong> Python, scikit-learn, imbalanced-learn, XGBoost, SHAP, LIME
            </p>
            <p>
              <strong className="text-white">Backend:</strong> FastAPI + Uvicorn + Pydantic
            </p>
            <p>
              <strong className="text-white">Frontend:</strong> React, Vite, Tailwind CSS, Recharts, React Router
            </p>
            <p className="pt-2 text-slate-400">
              Best Model: Logistic Regression (<code className="text-teal-300">models/best_model.pkl</code>) · Threshold: <code className="text-teal-300">0.41</code>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
