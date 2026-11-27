"""CardioSense Streamlit Application — 10-Year CHD Risk Assessment & Explainability.

Phase 5 (Polish):
- Tooltips on every sidebar input (help= parameter)
- Input validation: diastolic must be < systolic, cigs auto-capped for non-smokers
- "How to Read the SHAP Plot" collapsible guide
- Disclaimer always visible at bottom of Risk tab
- Model Performance tab: threshold explanation added
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import (
    FEATURE_ORDER,
    build_shap_explainer,
    compute_shap_values,
    explain_patient,
    preprocess_features,
)
from src.recommend import (
    LOW_RISK_MAX,
    MODERATE_RISK_MAX,
    classify_risk_level,
    generate_recommendations,
    get_risk_style,
)
from src.train import load_and_split_data

DATA_PATH  = PROJECT_ROOT / "data" / "framingham.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

EDUCATION_OPTIONS = {
    1: "1 — Some High School",
    2: "2 — High School / GED",
    3: "3 — Some College / Vocational",
    4: "4 — College Degree+",
}

# ── Sidebar input tooltips ─────────────────────────────────────────────────────
TOOLTIPS: dict[str, str] = {
    "age":             "Patient age in years (Framingham study range: 32–70).",
    "sex":             "Biological sex at birth. Sex is a known CHD risk modifier.",
    "education":       "Highest education level completed. Used as a socioeconomic proxy.",
    "currentSmoker":   "Check if the patient currently smokes any tobacco.",
    "cigsPerDay":      "Average number of cigarettes smoked per day. Set 0 for non-smokers.",
    "sysBP":           "Systolic blood pressure (mmHg). Elevated ≥ 130, Stage 2 ≥ 140.",
    "diaBP":           "Diastolic blood pressure (mmHg). Elevated ≥ 80, Stage 2 ≥ 90. "
                       "Must be lower than systolic BP.",
    "totChol":         "Total cholesterol (mg/dL). Borderline-high ≥ 200, high ≥ 240.",
    "BMI":             "Body Mass Index (kg/m²). Overweight ≥ 25, obese ≥ 30.",
    "heartRate":       "Resting heart rate in beats per minute.",
    "glucose":         "Fasting blood glucose (mg/dL). Impaired ≥ 100, diabetic range ≥ 126.",
    "prevalentHyp":    "Whether the patient has been previously diagnosed with hypertension.",
    "BPMeds":          "Whether the patient is currently taking blood-pressure-lowering medication.",
    "diabetes":        "Whether the patient has been previously diagnosed with diabetes mellitus.",
    "prevalentStroke": "Whether the patient has had a prior stroke.",
}


@st.cache_resource(show_spinner="Loading CardioSense model and SHAP explainer…")
def load_prediction_resources() -> tuple[object, dict, object]:
    """Load models/best_model.pkl, metadata, and SHAP LinearExplainer."""
    pipeline = joblib.load(MODELS_DIR / "best_model.pkl")
    meta_path = MODELS_DIR / "model_metadata.json"
    metadata = {"tuned_threshold": 0.41, "best_model": "Logistic Regression"}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    X_train, _, _, _ = load_and_split_data(DATA_PATH)
    shap_explainer, _ = build_shap_explainer(pipeline, X_train)
    return pipeline, metadata, shap_explainer


def draw_risk_gauge(probability: float, risk_level: str) -> plt.Figure:
    """Render a clean semi-circular probability gauge colored by risk tier."""
    style = get_risk_style(risk_level)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.set_aspect("equal")

    segments = [
        (0.0, LOW_RISK_MAX, "#38A169"),
        (LOW_RISK_MAX, MODERATE_RISK_MAX, "#D69E2E"),
        (MODERATE_RISK_MAX, 1.0, "#E53E3E"),
    ]
    for start_p, end_p, color in segments:
        theta2 = 180.0 * (1.0 - start_p)
        theta1 = 180.0 * (1.0 - end_p)
        wedge = patches.Wedge(
            center=(0, 0), r=1.0, theta1=theta1, theta2=theta2,
            width=0.28, facecolor=color, edgecolor="white", linewidth=2, alpha=0.90,
        )
        ax.add_patch(wedge)

    for tick_p in [0.0, LOW_RISK_MAX, MODERATE_RISK_MAX, 0.50, 0.75, 1.0]:
        ang = np.deg2rad(180.0 * (1.0 - tick_p))
        ax.text(
            1.14 * np.cos(ang), 1.14 * np.sin(ang),
            f"{int(tick_p * 100)}%", ha="center", va="center",
            fontsize=8.5, color="#4A5568",
            fontweight="bold" if tick_p in (LOW_RISK_MAX, MODERATE_RISK_MAX) else "normal",
        )

    needle_ang = np.deg2rad(180.0 * (1.0 - np.clip(probability, 0.0, 1.0)))
    ax.plot([0, 0.80 * np.cos(needle_ang)], [0, 0.80 * np.sin(needle_ang)],
            color="#1A202C", linewidth=3.0, zorder=5)
    ax.add_patch(patches.Circle((0, 0), 0.06, color="#1A202C", zorder=6))

    ax.text(0, 0.24, f"{probability * 100:.1f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=style["color"])
    ax.text(0, -0.14, f"{risk_level} Risk Tier", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#2D3748")

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-0.26, 1.28)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_shap_waterfall(shap_row: shap.Explanation, prob: float) -> plt.Figure:
    """Create a matplotlib figure containing the local SHAP waterfall plot."""
    plt.close("all")
    fig = plt.figure(figsize=(8.5, 5.2))
    shap.plots.waterfall(shap_row, max_display=10, show=False)
    plt.title(
        f"Local SHAP Waterfall (Predicted P = {prob:.1%})",
        fontsize=11.5, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    return plt.gcf()


def _shap_reading_guide() -> None:
    """Render a collapsible 'How to read the SHAP waterfall plot' guide."""
    with st.expander("ℹ️  How to Read the SHAP Waterfall Plot", expanded=False):
        st.markdown(
            """
**SHAP (SHapley Additive exPlanations)** values explain *why* the model gave this patient a
particular predicted probability by attributing each feature's contribution in log-odds units.

| Element | Meaning |
|---------|---------|
| **E[f(x)]** (bottom bar) | The model's average prediction across all training patients (baseline) |
| **f(x)** (top value) | The final log-odds output for *this* patient |
| **Red bars →** | Features that *increase* CHD risk (push log-odds higher) |
| **Blue bars ←** | Features that *decrease* CHD risk (push log-odds lower) |
| **Bar length** | Magnitude of that feature's contribution |

> **Example:** If *Age = 65* has a long red bar, it means the patient's age is a strong
> upward driver compared to the average patient in training data.

The bars stack from the baseline to reach the final predicted probability shown at the top.
            """
        )


def _validate_inputs(sys_bp: float, dia_bp: float, cigs: int, smoker: bool) -> list[str]:
    """Return a list of validation error messages (empty = all OK)."""
    errors: list[str] = []
    if dia_bp >= sys_bp:
        errors.append(
            f"⚠️ **Diastolic BP ({dia_bp:.1f} mmHg) ≥ Systolic BP ({sys_bp:.1f} mmHg).**  "
            "Diastolic pressure is always lower than systolic in a healthy reading. "
            "Please correct the values."
        )
    if not smoker and cigs > 0:
        errors.append(
            f"⚠️ **Cigarettes/Day = {cigs} but 'Current Smoker' is unchecked.**  "
            "If the patient smokes, tick 'Current Smoker'. If not, set Cigarettes/Day to 0."
        )
    return errors


def render_sidebar_form() -> tuple[pd.DataFrame, bool]:
    """Render the sidebar patient input form and return (patient_df, clicked)."""
    st.sidebar.title("CardioSense Inputs")
    st.sidebar.caption("Enter patient clinical measurements and medical history.")

    with st.sidebar.form("patient_input_form"):
        st.subheader("Demographics & Lifestyle")
        age = st.slider(
            "Age (years)", min_value=30, max_value=80, value=52, step=1,
            help=TOOLTIPS["age"],
        )
        male_label = st.selectbox(
            "Biological Sex",
            options=["Female (0)", "Male (1)"],
            index=1,
            help=TOOLTIPS["sex"],
        )
        male = 1 if "1" in male_label else 0
        education = st.selectbox(
            "Education Level",
            options=list(EDUCATION_OPTIONS.keys()),
            format_func=lambda k: EDUCATION_OPTIONS[k],
            index=1,
            help=TOOLTIPS["education"],
        )
        current_smoker = st.checkbox(
            "Current Smoker", value=False, help=TOOLTIPS["currentSmoker"],
        )
        cigs_per_day = st.number_input(
            "Cigarettes per Day",
            min_value=0, max_value=70, value=0, step=1,
            help=TOOLTIPS["cigsPerDay"],
        )

        st.subheader("Vitals & Laboratory Labs")
        sys_bp = st.number_input(
            "Systolic BP (mmHg)", min_value=80.0, max_value=295.0, value=134.0, step=1.0,
            help=TOOLTIPS["sysBP"],
        )
        dia_bp = st.number_input(
            "Diastolic BP (mmHg)", min_value=45.0, max_value=150.0, value=84.0, step=1.0,
            help=TOOLTIPS["diaBP"],
        )
        tot_chol = st.number_input(
            "Total Cholesterol (mg/dL)", min_value=100.0, max_value=700.0, value=238.0, step=1.0,
            help=TOOLTIPS["totChol"],
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=15.0, max_value=57.0, value=26.4, step=0.1,
            help=TOOLTIPS["BMI"],
        )
        heart_rate = st.number_input(
            "Heart Rate (bpm)", min_value=40.0, max_value=150.0, value=75.0, step=1.0,
            help=TOOLTIPS["heartRate"],
        )
        glucose = st.number_input(
            "Fasting Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=88.0, step=1.0,
            help=TOOLTIPS["glucose"],
        )

        st.subheader("Clinical History")
        prevalent_hyp    = st.checkbox("Hypertension History",  value=False, help=TOOLTIPS["prevalentHyp"])
        bp_meds          = st.checkbox("On BP Medication",       value=False, help=TOOLTIPS["BPMeds"])
        diabetes         = st.checkbox("Diabetes History",       value=False, help=TOOLTIPS["diabetes"])
        prevalent_stroke = st.checkbox("Prior Stroke History",   value=False, help=TOOLTIPS["prevalentStroke"])

        predict_clicked = st.form_submit_button("Predict", type="primary", width="stretch")

    patient_row = {
        "age":             float(age),
        "cigsPerDay":      float(cigs_per_day),
        "totChol":         float(tot_chol),
        "sysBP":           float(sys_bp),
        "diaBP":           float(dia_bp),
        "BMI":             float(bmi),
        "heartRate":       float(heart_rate),
        "glucose":         float(glucose),
        "male":            int(male),
        "education":       float(education),
        "currentSmoker":   int(current_smoker or cigs_per_day > 0),
        "BPMeds":          float(int(bp_meds)),
        "prevalentStroke": int(prevalent_stroke),
        "prevalentHyp":    int(prevalent_hyp),
        "diabetes":        int(diabetes),
    }
    patient_df = pd.DataFrame([patient_row])[FEATURE_ORDER]
    return patient_df, predict_clicked


def render_risk_tab(patient_df: pd.DataFrame, predict_clicked: bool) -> None:
    """Render the Risk Assessment, SHAP Explanation, and Recommendations tab."""

    # ── Input validation (shown before predict result) ────────────────────────
    if predict_clicked:
        sys_bp_val = float(patient_df["sysBP"].iloc[0])
        dia_bp_val = float(patient_df["diaBP"].iloc[0])
        cigs_val   = int(patient_df["cigsPerDay"].iloc[0])
        smoker_val = bool(patient_df["currentSmoker"].iloc[0])
        validation_errors = _validate_inputs(sys_bp_val, dia_bp_val, cigs_val, smoker_val)

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()  # halt — do not run prediction on invalid input

        pipeline, metadata, shap_explainer = load_prediction_resources()
        threshold  = float(metadata.get("tuned_threshold", 0.41))
        explanation = explain_patient(
            patient_df,
            pipeline=pipeline,
            shap_explainer=shap_explainer,
            threshold=threshold,
            top_k=5,
        )
        patient_proc = preprocess_features(pipeline, patient_df, readable_names=True)
        shap_row     = compute_shap_values(shap_explainer, patient_proc)[0]
        prob         = explanation["predicted_probability"]
        risk_level   = classify_risk_level(prob)
        recommendations = generate_recommendations(patient_df, explanation["top_features"])

        st.session_state["last_prediction"] = {
            "patient_df":   patient_df,
            "explanation":  explanation,
            "shap_row":     shap_row,
            "risk_level":   risk_level,
            "recommendations": recommendations,
            "model_name":   metadata.get("best_model", "Logistic Regression"),
        }

    if "last_prediction" not in st.session_state:
        st.info(
            "Adjust the patient's clinical values in the left sidebar and click **Predict** "
            "to load `models/best_model.pkl`, estimate 10-year CHD risk, inspect local SHAP "
            "explanations, and review personalised preventive recommendations."
        )
    else:
        state        = st.session_state["last_prediction"]
        explanation  = state["explanation"]
        prob         = explanation["predicted_probability"]
        threshold    = explanation["threshold"]
        risk_level   = state["risk_level"]
        style        = get_risk_style(risk_level)
        recommendations = state["recommendations"]

        # 1. Colored Risk Card + Gauge
        col_card, col_gauge = st.columns([1.25, 1.0])
        with col_card:
            screening_status = (
                f"Above tuned screening cutoff ({threshold:.0%}) — Preventive follow-up advised"
                if prob >= threshold
                else f"Below tuned screening cutoff ({threshold:.0%})"
            )
            st.markdown(
                f"""
                <div style="
                    background-color: {style['bg']};
                    border-left: 6px solid {style['border']};
                    border-radius: 8px;
                    padding: 1.25rem 1.4rem;
                    margin-top: 0.4rem;
                ">
                    <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.06em; color: {style['color']};">
                        {style['badge']}
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #1A202C; margin: 0.25rem 0;">
                        {prob * 100:.1f}% <span style="font-size: 1rem; font-weight: 500; color: #4A5568;">10-Year CHD Probability</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #2D3748; margin-bottom: 0.4rem;">
                        <strong>Model:</strong> {state['model_name']} &nbsp;|&nbsp;
                        <strong>Tier cutoffs:</strong> Low &lt;10%, Moderate 10–25%, High &gt;25%
                    </div>
                    <div style="font-size: 0.9rem; color: {style['color']}; font-weight: 600;">
                        {screening_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_gauge:
            fig_gauge = draw_risk_gauge(prob, risk_level)
            st.pyplot(fig_gauge, width="stretch")
            plt.close(fig_gauge)

        st.divider()

        # 2. SHAP Waterfall + "How to Read" guide + Top Contributing Factors
        st.subheader("Why Did the Model Predict This Risk? (Local SHAP Explanation)")
        _shap_reading_guide()

        col_wf, col_top = st.columns([1.35, 1.0])
        with col_wf:
            fig_wf = draw_shap_waterfall(state["shap_row"], prob)
            st.pyplot(fig_wf, width="stretch")
            plt.close(fig_wf)

        with col_top:
            st.markdown("#### Top 5 Contributing Features")
            for rank, item in enumerate(explanation["top_features"], start=1):
                is_inc = item["direction"] == "increases risk"
                arrow  = "▲" if is_inc else "▼"
                color  = "#C53030" if is_inc else "#2B6CB0"
                val_str = (
                    f"{item['value']:.2f}".rstrip("0").rstrip(".")
                    if item["value"] is not None else "N/A"
                )
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #E2E8F0;
                        border-left: 4px solid {color};
                        border-radius: 6px;
                        padding: 0.55rem 0.8rem;
                        margin-bottom: 0.5rem;
                        background-color: #FFFFFF;
                    ">
                        <div style="font-weight: 700; color: #1A202C;">
                            {rank}. {item['feature']} <span style="font-weight: 500; color: #4A5568;">(Value: {val_str})</span>
                        </div>
                        <div style="font-size: 0.88rem; color: {color}; font-weight: 600;">
                            {arrow} {item['direction'].title()} (SHAP: {item['shap_value']:+.4f} log-odds)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # 3. Personalized Preventive Recommendations
        st.subheader("Personalised Preventive Recommendations")
        if not recommendations:
            st.success(
                "No elevated modifiable risk factors detected across smoking, blood pressure, "
                "total cholesterol, BMI, or fasting glucose. Maintain healthy lifestyle habits "
                "and routine preventive checkups."
            )
        else:
            for rec in recommendations:
                driver_tag = " · Top SHAP Risk Driver" if rec["is_top_shap_driver"] else ""
                with st.expander(
                    f"[{rec['severity']}] {rec['category']} — Observed: {rec['observed']}{driver_tag}",
                    expanded=True,
                ):
                    st.markdown(f"**Guideline Target:** {rec['guideline_target']}")
                    st.write(rec["recommendation"])

    # ── Disclaimer — always rendered at the bottom ────────────────────────────
    st.divider()
    st.warning(
        "**Medical Disclaimer:** CardioSense is an academic decision-support prototype built on "
        "the Framingham Heart Study dataset. Predictions and recommendations are for educational "
        "purposes only and do **not** constitute medical advice, diagnosis, or treatment. Always "
        "consult a qualified healthcare professional before making clinical decisions."
    )


def render_performance_tab() -> None:
    """Display model comparison metrics and saved ROC/confusion/threshold plots."""
    st.subheader("Test-Set Model Comparison (Stratified 80/20 Split)")

    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, width="stretch", hide_index=True)
    else:
        st.error(f"Comparison table not found at {comp_path}. Run `python src/train.py` first.")

    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        meta  = json.loads(meta_path.read_text(encoding="utf-8"))
        tuned = meta.get("test_metrics_at_tuned_threshold", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Pipeline",  meta.get("best_model", "Logistic Regression"))
        c2.metric("Test ROC-AUC",       f"{tuned.get('roc_auc', 0.0):.4f}")
        c3.metric("Tuned Threshold",    f"{meta.get('tuned_threshold', 0.41):.2f}")
        c4.metric("Tuned CHD Recall",   f"{tuned.get('recall', 0.0):.2%}")

    st.info(
        "**Decision Threshold:** The default classification threshold of 0.50 was lowered to "
        f"**0.41** by maximising CHD recall on out-of-fold (OOF) training predictions. "
        "A lower threshold flags more patients as high-risk, trading some precision for higher "
        "sensitivity — appropriate for a screening tool where missing a true CHD case is costly."
    )

    roc_path = REPORTS_DIR / "roc_curves_comparison.png"
    cm_path  = REPORTS_DIR / "confusion_matrices.png"
    thr_path = REPORTS_DIR / "threshold_tuning.png"

    col_left, col_right = st.columns(2)
    with col_left:
        if roc_path.exists():
            st.image(str(roc_path), caption="Held-Out Test ROC Curves", width="stretch")
    with col_right:
        if thr_path.exists():
            st.image(str(thr_path), caption="Training OOF Decision Threshold Tuning", width="stretch")

    if cm_path.exists():
        st.image(str(cm_path), caption="Held-Out Test Confusion Matrices (Default t = 0.50)", width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="CardioSense — Explainable 10-Year CHD Risk",
        page_icon="❤️",
        layout="wide",
    )

    st.title("CardioSense: Explainable 10-Year Coronary Heart Disease Risk")
    st.caption("Leakage-free ML risk stratification · SHAP local explanations · Personalised preventive guidance.")

    patient_df, predict_clicked = render_sidebar_form()

    tab_risk, tab_perf = st.tabs(["Risk Assessment & Explainability", "Model Performance"])
    with tab_risk:
        render_risk_tab(patient_df, predict_clicked)
    with tab_perf:
        render_performance_tab()


if __name__ == "__main__":
    main()
