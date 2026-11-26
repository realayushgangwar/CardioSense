"""Global and local explainability (SHAP & LIME) for CardioSense.

Loads the trained pipeline from ``models/best_model.pkl``, computes exact
SHAP explanations on preprocessed features using the model-appropriate
explainer (``LinearExplainer`` for Logistic Regression, ``TreeExplainer`` for
tree ensembles), generates LIME tabular explanations, and provides the
reusable ``explain_patient(patient_df)`` helper for downstream inference/UI.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    load_and_split_data,
)

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

READABLE_FEATURE_NAMES: dict[str, str] = {
    "age": "Age (years)",
    "cigsPerDay": "Cigarettes / Day",
    "totChol": "Total Cholesterol (mg/dL)",
    "sysBP": "Systolic BP (mmHg)",
    "diaBP": "Diastolic BP (mmHg)",
    "BMI": "BMI (kg/m²)",
    "heartRate": "Heart Rate (bpm)",
    "glucose": "Glucose (mg/dL)",
    "male": "Sex (Male = 1)",
    "education": "Education Level (1–4)",
    "currentSmoker": "Current Smoker (1 = Yes)",
    "BPMeds": "BP Medication (1 = Yes)",
    "prevalentStroke": "Prevalent Stroke (1 = Yes)",
    "prevalentHyp": "Prevalent Hypertension (1 = Yes)",
    "diabetes": "Diabetes (1 = Yes)",
}

READABLE_LIST = [READABLE_FEATURE_NAMES[c] for c in FEATURE_ORDER]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_model_and_metadata(
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the saved pipeline from models/best_model.pkl and its metadata."""
    root = _project_root()
    mdir = models_dir or (root / "models")
    pipeline = joblib.load(mdir / "best_model.pkl")

    metadata_path = mdir / "model_metadata.json"
    metadata: dict[str, Any] = {"tuned_threshold": 0.50}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def preprocess_features(
    pipeline: Any,
    X: pd.DataFrame,
    readable_names: bool = True,
) -> pd.DataFrame:
    """Apply the pipeline's fitted ColumnTransformer and return a DataFrame."""
    ordered_X = X[FEATURE_ORDER]
    transformed = pipeline.named_steps["preprocessor"].transform(ordered_X)
    cols = READABLE_LIST if readable_names else FEATURE_ORDER
    return pd.DataFrame(transformed, columns=cols, index=X.index)


def build_shap_explainer(
    pipeline: Any,
    X_train: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    """Build the model-appropriate SHAP explainer on preprocessed training data."""
    X_train_proc = preprocess_features(pipeline, X_train, readable_names=True)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "coef_"):
        masker = shap.maskers.Independent(
            X_train_proc, max_samples=len(X_train_proc)
        )
        explainer = shap.LinearExplainer(classifier, masker)
    elif classifier.__class__.__name__ in {
        "RandomForestClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
    }:
        explainer = shap.TreeExplainer(classifier, data=X_train_proc)
    else:
        explainer = shap.Explainer(classifier.predict_proba, X_train_proc)

    return explainer, X_train_proc


def compute_shap_values(explainer: Any, X_proc: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP Explanation objects with readable feature names."""
    shap_values = explainer(X_proc)
    # Normalize multi-output tree/kernel explanations to positive CHD class (index 1)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap_values.feature_names = list(X_proc.columns)
    return shap_values


def build_lime_explainer(
    pipeline: Any,
    X_train: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[lime.lime_tabular.LimeTabularExplainer, pd.DataFrame]:
    """Create a LimeTabularExplainer on preprocessed training features."""
    X_train_proc = preprocess_features(pipeline, X_train, readable_names=True)
    categorical_indices = list(range(len(NUMERIC_FEATURES), len(FEATURE_ORDER)))

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_proc.values,
        feature_names=READABLE_LIST,
        class_names=["No CHD (0)", "10-Year CHD (1)"],
        categorical_features=categorical_indices,
        mode="classification",
        random_state=random_state,
    )
    return lime_explainer, X_train_proc


@lru_cache(maxsize=1)
def _get_default_explain_context() -> tuple[Any, dict[str, Any], Any]:
    """Cache default model, metadata, and SHAP explainer for explain_patient()."""
    root = _project_root()
    pipeline, metadata = load_model_and_metadata(root / "models")
    X_train, _, _, _ = load_and_split_data(root / "data" / "framingham.csv")
    shap_explainer, _ = build_shap_explainer(pipeline, X_train)
    return pipeline, metadata, shap_explainer


def explain_patient(
    patient_df: pd.DataFrame | pd.Series,
    pipeline: Any | None = None,
    shap_explainer: Any | None = None,
    threshold: float | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Explain a single patient's 10-year CHD risk prediction using SHAP.

    Parameters
    ----------
    patient_df:
        Single-row DataFrame (or Series) containing the 15 raw clinical features.
    pipeline:
        Optional pre-loaded fitted ``imblearn`` Pipeline. Uses ``models/best_model.pkl`` if omitted.
    shap_explainer:
        Optional pre-built SHAP explainer. Built from training background if omitted.
    threshold:
        Optional classification cutoff. Defaults to ``tuned_threshold`` in ``model_metadata.json``.
    top_k:
        Number of top contributing features to return (default: 5).

    Returns
    -------
    dict
        Predicted probability, binary prediction at threshold, and top contributing
        features with signed SHAP value and risk direction (``increases risk`` or
        ``decreases risk``).
    """
    if isinstance(patient_df, pd.Series):
        patient_df = patient_df.to_frame().T

    if len(patient_df) != 1:
        raise ValueError(f"explain_patient expects exactly 1 row, got {len(patient_df)}.")

    if pipeline is None or shap_explainer is None or threshold is None:
        default_pipe, default_meta, default_exp = _get_default_explain_context()
        pipeline = pipeline or default_pipe
        shap_explainer = shap_explainer or default_exp
        if threshold is None:
            threshold = float(default_meta.get("tuned_threshold", 0.50))

    ordered_df = patient_df[FEATURE_ORDER]
    prob = float(pipeline.predict_proba(ordered_df)[:, 1][0])

    patient_proc = preprocess_features(pipeline, ordered_df, readable_names=True)
    sv = compute_shap_values(shap_explainer, patient_proc)
    shap_row = sv.values[0]

    contributions = []
    for raw_col, readable_name, shap_val in zip(FEATURE_ORDER, READABLE_LIST, shap_row):
        raw_val = ordered_df.iloc[0][raw_col]
        contributions.append(
            {
                "feature": readable_name,
                "raw_feature": raw_col,
                "shap_value": round(float(shap_val), 4),
                "abs_shap_value": round(float(abs(shap_val)), 4),
                "direction": "increases risk" if shap_val >= 0 else "decreases risk",
                "value": None if pd.isna(raw_val) else float(raw_val),
            }
        )

    contributions.sort(key=lambda item: item["abs_shap_value"], reverse=True)

    return {
        "predicted_probability": round(prob, 4),
        "predicted_label": int(prob >= threshold),
        "threshold": round(float(threshold), 3),
        "top_features": contributions[:top_k],
    }


def plot_shap_beeswarm(
    shap_values: shap.Explanation,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generate and optionally save a global SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 6.5))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(
        "Global SHAP Summary (Beeswarm) — Feature Impact on 10-Year CHD Log-Odds",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    fig = plt.gcf()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_bar(
    shap_values: shap.Explanation,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generate and optionally save a global bar plot of mean |SHAP| values."""
    plt.figure(figsize=(9.5, 6.0))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title(
        "Global Feature Importance — Mean |SHAP Value| Across Held-Out Test Set",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    fig = plt.gcf()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_waterfall(
    shap_explanation_row: shap.Explanation,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Generate and optionally save a local SHAP waterfall plot for one patient."""
    plt.figure(figsize=(9.5, 6.0))
    shap.plots.waterfall(shap_explanation_row, max_display=12, show=False)
    plt.title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_lime_comparison(
    lime_high: Any,
    lime_low: Any,
    high_title: str,
    low_title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot side-by-side LIME feature weights for high-risk and low-risk patients."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    for ax, exp_obj, title in zip(axes, [lime_high, lime_low], [high_title, low_title]):
        items = exp_obj.as_list()
        # Sort by absolute contribution ascending so largest bar sits at top
        items_sorted = sorted(items, key=lambda x: abs(x[1]))
        labels = [k for k, _ in items_sorted]
        weights = [v for _, v in items_sorted]
        colors = ["#D94848" if w >= 0 else "#2B6CB0" for w in weights]

        ax.barh(labels, weights, color=colors, edgecolor="#222222", linewidth=0.5)
        ax.axvline(0, color="#1A202C", linewidth=1.0)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("LIME Local Linear Weight (Red = Increases Risk, Blue = Decreases Risk)")
        ax.grid(axis="x", alpha=0.35)

    fig.suptitle(
        "LIME Local Explanations (Top Features on Preprocessed Feature Space)",
        fontsize=13.5,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def select_representative_patients(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[int, int]:
    """Pick one true high-risk patient and one true low-risk patient from X_test."""
    probs = pipeline.predict_proba(X_test[FEATURE_ORDER])[:, 1]
    high_risk_pos = int(np.argmax(probs * (y_test.values == 1)))
    low_risk_pos = int(np.argmin(probs + (y_test.values == 1) * 10.0))
    return high_risk_pos, low_risk_pos


def main() -> None:
    """Generate and save all SHAP and LIME explainability plots to reports/."""
    import matplotlib
    matplotlib.use("Agg")

    root = _project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline, metadata = load_model_and_metadata(root / "models")
    threshold = float(metadata.get("tuned_threshold", 0.41))
    X_train, X_test, _, y_test = load_and_split_data(root / "data" / "framingham.csv")

    shap_explainer, _ = build_shap_explainer(pipeline, X_train)
    X_test_proc = preprocess_features(pipeline, X_test, readable_names=True)
    shap_values = compute_shap_values(shap_explainer, X_test_proc)

    fig_bee = plot_shap_beeswarm(shap_values, save_path=reports_dir / "shap_summary_beeswarm.png")
    plt.close(fig_bee)
    fig_bar = plot_shap_bar(shap_values, save_path=reports_dir / "shap_feature_importance_bar.png")
    plt.close(fig_bar)

    high_pos, low_pos = select_representative_patients(pipeline, X_test, y_test)
    high_exp = explain_patient(X_test.iloc[[high_pos]], pipeline=pipeline, shap_explainer=shap_explainer, threshold=threshold)
    low_exp = explain_patient(X_test.iloc[[low_pos]], pipeline=pipeline, shap_explainer=shap_explainer, threshold=threshold)

    fig_wh = plot_shap_waterfall(
        shap_values[high_pos],
        title=f"Local SHAP Waterfall — High-Risk Patient (P={high_exp['predicted_probability']:.2%}, Actual=1)",
        save_path=reports_dir / "shap_waterfall_high_risk.png",
    )
    plt.close(fig_wh)
    fig_wl = plot_shap_waterfall(
        shap_values[low_pos],
        title=f"Local SHAP Waterfall — Low-Risk Patient (P={low_exp['predicted_probability']:.2%}, Actual=0)",
        save_path=reports_dir / "shap_waterfall_low_risk.png",
    )
    plt.close(fig_wl)

    lime_explainer, _ = build_lime_explainer(pipeline, X_train)
    clf = pipeline.named_steps["classifier"]
    lime_high = lime_explainer.explain_instance(
        X_test_proc.iloc[high_pos].values,
        clf.predict_proba,
        num_features=8,
    )
    lime_low = lime_explainer.explain_instance(
        X_test_proc.iloc[low_pos].values,
        clf.predict_proba,
        num_features=8,
    )
    fig_lime = plot_lime_comparison(
        lime_high,
        lime_low,
        high_title=f"High-Risk Patient (P={high_exp['predicted_probability']:.2%}, Actual=1)",
        low_title=f"Low-Risk Patient (P={low_exp['predicted_probability']:.2%}, Actual=0)",
        save_path=reports_dir / "lime_local_explanations.png",
    )
    plt.close(fig_lime)

    print("High-risk patient explanation:", json.dumps(high_exp, indent=2))
    print("Low-risk patient explanation:", json.dumps(low_exp, indent=2))
    print("Saved all SHAP and LIME plots to reports/.")


if __name__ == "__main__":
    main()
