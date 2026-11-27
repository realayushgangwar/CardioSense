"""CardioSense Streamlit Application — Explainable 10-Year CHD Risk Dashboard."""

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

DATA_PATH = PROJECT_ROOT / "data" / "framingham.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

EDUCATION_OPTIONS = {
    1: "1 — Some High School",
    2: "2 — High School / GED",
    3: "3 — Some College / Vocational",
    4: "4 — College Degree+",
}

CATEGORY_ICONS: dict[str, str] = {
    "Smoking Cessation": ":material/smoke_free:",
    "Blood Pressure Control": ":material/monitor_heart:",
    "Lipid & Cholesterol Management": ":material/bloodtype:",
    "Weight & Physical Activity (BMI)": ":material/directions_run:",
    "Glycemic & Metabolic Control": ":material/glucose:",
}

TOOLTIPS: dict[str, str] = {
    "age": "Patient age in years (Framingham cohort range: 32–70).",
    "sex": "Biological sex at birth; male sex is associated with higher baseline CHD risk.",
    "education": "Highest education level completed (1 = Some High School to 4 = College+).",
    "currentSmoker": "Check if the patient currently smokes tobacco.",
    "cigsPerDay": "Average cigarettes smoked per day (0 for non-smokers).",
    "sysBP": "Systolic blood pressure in mmHg (Elevated ≥ 130, Stage 2 ≥ 140).",
    "diaBP": "Diastolic blood pressure in mmHg (Elevated ≥ 80, Stage 2 ≥ 90). Must be below Systolic BP.",
    "totChol": "Total serum cholesterol in mg/dL (Borderline ≥ 200, High ≥ 240).",
    "BMI": "Body Mass Index in kg/m² (Overweight ≥ 25, Obese ≥ 30).",
    "heartRate": "Resting heart rate in beats per minute (bpm).",
    "glucose": "Fasting blood glucose in mg/dL (Impaired ≥ 100, Diabetic range ≥ 126).",
    "prevalentHyp": "Prior clinical diagnosis of hypertension.",
    "BPMeds": "Currently prescribed blood-pressure-lowering medication.",
    "diabetes": "Prior clinical diagnosis of diabetes mellitus.",
    "prevalentStroke": "History of prior stroke.",
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
    """Render a semi-circular probability gauge colored by risk tier."""
    style = get_risk_style(risk_level)
    fig, ax = plt.subplots(figsize=(5.0, 2.85))
    ax.set_aspect("equal")

    segments = [
        (0.0, LOW_RISK_MAX, "#16A34A"),
        (LOW_RISK_MAX, MODERATE_RISK_MAX, "#D97706"),
        (MODERATE_RISK_MAX, 1.0, "#DC2626"),
    ]
    for start_p, end_p, color in segments:
        theta2 = 180.0 * (1.0 - start_p)
        theta1 = 180.0 * (1.0 - end_p)
        wedge = patches.Wedge(
            center=(0, 0),
            r=1.0,
            theta1=theta1,
            theta2=theta2,
            width=0.27,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.92,
        )
        ax.add_patch(wedge)

    for tick_p in [0.0, LOW_RISK_MAX, MODERATE_RISK_MAX, 0.50, 0.75, 1.0]:
        ang = np.deg2rad(180.0 * (1.0 - tick_p))
        ax.text(
            1.14 * np.cos(ang),
            1.14 * np.sin(ang),
            f"{int(tick_p * 100)}%",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#475569",
            fontweight="bold" if tick_p in (LOW_RISK_MAX, MODERATE_RISK_MAX) else "normal",
        )

    needle_ang = np.deg2rad(180.0 * (1.0 - np.clip(probability, 0.0, 1.0)))
    ax.plot(
        [0, 0.80 * np.cos(needle_ang)],
        [0, 0.80 * np.sin(needle_ang)],
        color="#0F172A",
        linewidth=3.0,
        zorder=5,
    )
    ax.add_patch(patches.Circle((0, 0), 0.06, color="#0F172A", zorder=6))

    ax.text(
        0,
        0.24,
        f"{probability * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=style["color"],
    )
    ax.text(
        0,
        -0.14,
        f"{risk_level} Risk Tier",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1E293B",
    )

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-0.26, 1.28)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_shap_waterfall(shap_row: shap.Explanation, prob: float) -> plt.Figure:
    """Create a matplotlib figure containing the local SHAP waterfall plot."""
    plt.close("all")
    plt.figure(figsize=(8.2, 5.0))
    shap.plots.waterfall(shap_row, max_display=10, show=False)
    plt.title(
        f"Local SHAP Waterfall (Predicted P = {prob:.1%})",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    return plt.gcf()


def _validate_inputs(
    sys_bp: float,
    dia_bp: float,
    cigs: int,
    smoker: bool,
) -> list[str]:
    """Return friendly validation error messages (empty list means valid)."""
    errors: list[str] = []
    if dia_bp >= sys_bp:
        errors.append(
            f"Diastolic BP ({dia_bp:.0f} mmHg) must be lower than Systolic BP ({sys_bp:.0f} mmHg). "
            "Please adjust the blood pressure readings before predicting."
        )
    if not smoker and cigs > 0:
        errors.append(
            f"Cigarettes per Day is set to {cigs}, but 'Current Smoker' is unchecked. "
            "Either check 'Current Smoker' or set Cigarettes per Day to 0."
        )
    return errors


def render_sidebar_form() -> tuple[pd.DataFrame, bool]:
    """Render sidebar inputs organized into Demographics, Lifestyle, and Clinical Measurements."""
    st.sidebar.markdown("## :material/favorite: Patient Profile")
    st.sidebar.caption("Complete the sections below and click **Predict**.")

    with st.sidebar.form("patient_input_form"):
        st.markdown("### Demographics")
        age = st.slider(
            "Age (years)",
            min_value=30,
            max_value=80,
            value=52,
            step=1,
            help=TOOLTIPS["age"],
        )
        male_label = st.selectbox(
            "Biological sex",
            options=["Female (0)", "Male (1)"],
            index=1,
            help=TOOLTIPS["sex"],
        )
        male = 1 if "1" in male_label else 0
        education = st.selectbox(
            "Education level",
            options=list(EDUCATION_OPTIONS.keys()),
            format_func=lambda k: EDUCATION_OPTIONS[k],
            index=1,
            help=TOOLTIPS["education"],
        )

        st.divider()
        st.markdown("### Lifestyle")
        current_smoker = st.checkbox(
            "Current smoker",
            value=False,
            help=TOOLTIPS["currentSmoker"],
        )
        cigs_per_day = st.number_input(
            "Cigarettes per day",
            min_value=0,
            max_value=70,
            value=0,
            step=1,
            help=TOOLTIPS["cigsPerDay"],
        )

        st.divider()
        st.markdown("### Clinical Measurements")
        sys_bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=80.0,
            max_value=295.0,
            value=134.0,
            step=1.0,
            help=TOOLTIPS["sysBP"],
        )
        dia_bp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=45.0,
            max_value=150.0,
            value=84.0,
            step=1.0,
            help=TOOLTIPS["diaBP"],
        )
        tot_chol = st.number_input(
            "Total cholesterol (mg/dL)",
            min_value=100.0,
            max_value=700.0,
            value=238.0,
            step=1.0,
            help=TOOLTIPS["totChol"],
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=15.0,
            max_value=57.0,
            value=26.4,
            step=0.1,
            help=TOOLTIPS["BMI"],
        )
        heart_rate = st.number_input(
            "Resting heart rate (bpm)",
            min_value=40.0,
            max_value=150.0,
            value=75.0,
            step=1.0,
            help=TOOLTIPS["heartRate"],
        )
        glucose = st.number_input(
            "Fasting glucose (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=88.0,
            step=1.0,
            help=TOOLTIPS["glucose"],
        )

        st.caption("Medical History")
        prevalent_hyp = st.checkbox("Hypertension history", value=False, help=TOOLTIPS["prevalentHyp"])
        bp_meds = st.checkbox("On BP medication", value=False, help=TOOLTIPS["BPMeds"])
        diabetes = st.checkbox("Diabetes history", value=False, help=TOOLTIPS["diabetes"])
        prevalent_stroke = st.checkbox("Prior stroke history", value=False, help=TOOLTIPS["prevalentStroke"])

        predict_clicked = st.form_submit_button(
            "Predict 10-Year CHD Risk",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

    patient_row = {
        "age": float(age),
        "cigsPerDay": float(cigs_per_day),
        "totChol": float(tot_chol),
        "sysBP": float(sys_bp),
        "diaBP": float(dia_bp),
        "BMI": float(bmi),
        "heartRate": float(heart_rate),
        "glucose": float(glucose),
        "male": int(male),
        "education": float(education),
        "currentSmoker": int(current_smoker),
        "BPMeds": float(int(bp_meds)),
        "prevalentStroke": int(prevalent_stroke),
        "prevalentHyp": int(prevalent_hyp),
        "diabetes": int(diabetes),
    }
    patient_df = pd.DataFrame([patient_row])[FEATURE_ORDER]
    return patient_df, predict_clicked


def render_prediction_tab(patient_df: pd.DataFrame, predict_clicked: bool) -> None:
    """Render the 3-part Prediction view: Risk Card + Gauge, Key Risk Factors, Recommendations."""
    if predict_clicked:
        sys_bp_val = float(patient_df["sysBP"].iloc[0])
        dia_bp_val = float(patient_df["diaBP"].iloc[0])
        cigs_val = int(patient_df["cigsPerDay"].iloc[0])
        smoker_val = bool(patient_df["currentSmoker"].iloc[0])
        validation_errors = _validate_inputs(sys_bp_val, dia_bp_val, cigs_val, smoker_val)

        if validation_errors:
            for err in validation_errors:
                st.error(err, title="Please check patient inputs", icon=":material/error:")
            return

        with st.spinner("Computing 10-year CHD risk and local SHAP attributions…"):
            pipeline, metadata, shap_explainer = load_prediction_resources()
            threshold = float(metadata.get("tuned_threshold", 0.41))
            explanation = explain_patient(
                patient_df,
                pipeline=pipeline,
                shap_explainer=shap_explainer,
                threshold=threshold,
                top_k=5,
            )
            patient_proc = preprocess_features(pipeline, patient_df, readable_names=True)
            shap_row = compute_shap_values(shap_explainer, patient_proc)[0]
            prob = explanation["predicted_probability"]
            risk_level = classify_risk_level(prob)
            recommendations = generate_recommendations(patient_df, explanation["top_features"])

            st.session_state["last_prediction"] = {
                "patient_df": patient_df,
                "explanation": explanation,
                "shap_row": shap_row,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "model_name": metadata.get("best_model", "Logistic Regression"),
            }

    if "last_prediction" not in st.session_state:
        st.info(
            "Enter patient demographics, lifestyle habits, and clinical measurements in the sidebar, "
            "then click **Predict 10-Year CHD Risk** to view risk stratification, SHAP drivers, and "
            "personalized preventive recommendations.",
            title="Ready for Risk Assessment",
            icon=":material/clinical_notes:",
        )
        return

    state = st.session_state["last_prediction"]
    explanation = state["explanation"]
    prob = explanation["predicted_probability"]
    threshold = explanation["threshold"]
    risk_level = state["risk_level"]
    style = get_risk_style(risk_level)
    recommendations = state["recommendations"]

    # ── Part 1: Large Colored Risk Card + Gauge ───────────────────────────────
    with st.container(border=True):
        col_card, col_gauge = st.columns([1.25, 1.0], vertical_alignment="center")
        with col_card:
            screening_status = (
                f"Above tuned screening cutoff ({threshold:.0%}) — Preventive follow-up advised"
                if prob >= threshold
                else f"Below tuned screening cutoff ({threshold:.0%}) — Routine monitoring"
            )
            st.markdown(
                f"""
                <div style="
                    background-color: {style['bg']};
                    border-left: 6px solid {style['border']};
                    border-radius: 10px;
                    padding: 1.35rem 1.5rem;
                ">
                    <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; color: {style['color']};">
                        {style['badge']}
                    </div>
                    <div style="font-size: 2.35rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">
                        {prob * 100:.1f}% <span style="font-size: 1rem; font-weight: 500; color: #475569;">10-Year CHD Risk</span>
                    </div>
                    <div style="font-size: 0.93rem; color: #334155; margin-bottom: 0.45rem;">
                        <strong>Model:</strong> {state['model_name']} &nbsp;|&nbsp;
                        <strong>Tiers:</strong> Low &lt;10% · Moderate 10–25% · High &gt;25%
                    </div>
                    <div style="font-size: 0.92rem; color: {style['color']}; font-weight: 600;">
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

    # ── Part 2: Key Risk Factors (SHAP Plot + Top 5 Drivers) ──────────────────
    with st.container(border=True):
        st.subheader(":material/bar_chart: Key Risk Factors")
        with st.expander("How to read the SHAP waterfall plot", expanded=False):
            st.markdown(
                "SHAP shows how each measurement shifts the prediction from the population baseline "
                "`E[f(X)]` to this patient's log-odds `f(x)`. **Red bars** push CHD risk higher; "
                "**blue bars** push CHD risk lower."
            )

        col_wf, col_top = st.columns([1.35, 1.0])
        with col_wf:
            fig_wf = draw_shap_waterfall(state["shap_row"], prob)
            st.pyplot(fig_wf, width="stretch")
            plt.close(fig_wf)

        with col_top:
            st.markdown("#### Top 5 Contributing Features")
            for rank, item in enumerate(explanation["top_features"], start=1):
                is_inc = item["direction"] == "increases risk"
                arrow = "▲" if is_inc else "▼"
                accent = "#DC2626" if is_inc else "#1D4ED8"
                val_str = (
                    f"{item['value']:.2f}".rstrip("0").rstrip(".")
                    if item["value"] is not None
                    else "N/A"
                )
                with st.container(border=True):
                    st.markdown(
                        f"**{rank}. {item['feature']}** (`{val_str}`)  \n"
                        f"<span style='color:{accent}; font-weight:600;'>"
                        f"{arrow} {item['direction'].title()} ({item['shap_value']:+.4f} log-odds)"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

    # ── Part 3: Recommendations Panel (Icons + Short Cards) ───────────────────
    with st.container(border=True):
        st.subheader(":material/health_and_safety: Recommendations")
        if not recommendations:
            st.success(
                "No elevated modifiable risk factors detected across smoking, blood pressure, "
                "cholesterol, BMI, or fasting glucose. Continue healthy habits and routine checkups.",
                title="All Modifiable Targets Within Guideline Range",
                icon=":material/check_circle:",
            )
        else:
            rec_cols = st.columns(2)
            for idx, rec in enumerate(recommendations):
                icon = CATEGORY_ICONS.get(rec["category"], ":material/medical_services:")
                driver_badge = " · **Top SHAP Driver**" if rec["is_top_shap_driver"] else ""
                with rec_cols[idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"#### {icon} {rec['category']}")
                        st.caption(
                            f"Severity: **{rec['severity']}** · Observed: **{rec['observed']}**{driver_badge}"
                        )
                        st.markdown(f"**Target:** {rec['guideline_target']}")
                        st.write(rec["recommendation"])


def render_performance_tab() -> None:
    """Display model comparison metrics and saved ROC/confusion/threshold plots."""
    st.subheader(":material/query_stats: Test-Set Model Comparison (Stratified 80/20 Split)")

    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, width="stretch", hide_index=True)
    else:
        st.error(f"Comparison table not found at {comp_path}. Run `python src/train.py` first.")

    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tuned = meta.get("test_metrics_at_tuned_threshold", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Pipeline", meta.get("best_model", "Logistic Regression"))
        c2.metric("Test ROC-AUC", f"{tuned.get('roc_auc', 0.0):.4f}")
        c3.metric("Tuned Threshold", f"{meta.get('tuned_threshold', 0.41):.2f}")
        c4.metric("Tuned CHD Recall", f"{tuned.get('recall', 0.0):.2%}")

    st.info(
        "The classification threshold was tuned from 0.50 down to **0.41** on training out-of-fold "
        "predictions to raise CHD screening recall from 60.47% to 74.42% (precision 21.52%), reducing "
        "missed high-risk patients.",
        title="Why Threshold t = 0.41?",
        icon=":material/tune:",
    )

    roc_path = REPORTS_DIR / "roc_curves_comparison.png"
    cm_path = REPORTS_DIR / "confusion_matrices.png"
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


def render_about_tab() -> None:
    """Display dataset, methodology, and limitations information."""
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader(":material/dataset: Dataset")
            st.markdown(
                """
                - **Source:** Framingham Heart Study (`data/framingham.csv`)
                - **Size:** 4,238 patients, 15 clinical/demographic predictors, 1 binary target (`TenYearCHD`)
                - **Class Balance:** 3,594 No CHD (84.8%) vs. 644 10-Year CHD (15.2%) — a 5.58 : 1 imbalance ratio
                - **Missing Data:** 645 missing cells across 7 columns (led by `glucose` at 9.16% and `education` at 2.48%), imputed inside the training pipeline
                """
            )

        with st.container(border=True):
            st.subheader(":material/account_tree: Methodology")
            st.markdown(
                """
                1. **Leakage-Free Split:** Stratified 80/20 split (`RANDOM_STATE = 42`) yielding 3,390 train and 848 test rows before any preprocessing.
                2. **Pipeline:** `ColumnTransformer` (median + `StandardScaler` for numeric; mode for categorical) followed by **SMOTE** inside `imblearn.pipeline.Pipeline` (applied to training folds only).
                3. **Models:** Logistic Regression, Random Forest, and XGBoost tuned via 5-fold stratified CV; Logistic Regression selected (`Test ROC-AUC = 0.6985`).
                4. **Explainability:** Exact `shap.LinearExplainer` attributions paired with ACC/AHA & ADA preventive guidelines.
                """
            )

    with col2:
        with st.container(border=True):
            st.subheader(":material/warning: Limitations & Ethical Considerations")
            st.markdown(
                """
                - **Single-Population Cohort:** Framingham participants were predominantly white residents of Framingham, Massachusetts; risk weights may not generalize directly to all contemporary or diverse populations.
                - **Not a Diagnostic Tool:** CardioSense is an academic decision-support prototype and must never replace physician evaluation or laboratory confirmation.
                - **Class Imbalance & Precision:** Tuning for 74.42% CHD recall produces 21.52% precision; positive screening flags should prompt non-invasive clinical review rather than alarm.
                - **SMOTE Probability Shift:** Oversampling shifts raw predicted probabilities upward relative to the 15.2% population base rate; post-hoc calibration is planned for future work.
                """
            )


def main() -> None:
    st.set_page_config(
        page_title="CardioSense — Explainable 10-Year CHD Risk",
        page_icon="❤️",
        layout="wide",
    )

    # Header with heart icon, CardioSense title, and one-line tagline
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0F294A 0%, #1D4ED8 100%);
            color: #FFFFFF;
            padding: 1.15rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.65rem; font-weight: 800; letter-spacing: -0.01em;">
                ❤️ CardioSense
            </div>
            <div style="font-size: 0.98rem; color: #E2E8F0; margin-top: 0.2rem;">
                Explainable machine learning for early 10-year coronary heart disease risk prediction and personalized prevention.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    patient_df, predict_clicked = render_sidebar_form()

    tab_pred, tab_perf, tab_about = st.tabs([
        ":material/favorite: Prediction",
        ":material/query_stats: Model Performance",
        ":material/info: About",
    ])
    with tab_pred:
        render_prediction_tab(patient_df, predict_clicked)
    with tab_perf:
        render_performance_tab()
    with tab_about:
        render_about_tab()

    # Persistent footer disclaimer
    st.divider()
    st.caption("CardioSense · Decision-support prototype, not medical advice.")


if __name__ == "__main__":
    main()
