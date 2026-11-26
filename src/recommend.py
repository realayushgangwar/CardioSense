"""Risk stratification and clinical guideline recommendations for CardioSense.

Converts predicted 10-year CHD probabilities into Low / Moderate / High risk
tiers and produces personalized preventive recommendations strictly for
modifiable risk factors that are elevated or risky for the given patient.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

# Probability cutoffs for 3-tier risk stratification
LOW_RISK_MAX = 0.10
MODERATE_RISK_MAX = 0.25

# Standard clinical guideline thresholds for modifiable CHD risk factors
SMOKING_CIGS_THRESHOLD = 1.0
SYS_BP_ELEVATED_THRESHOLD = 130.0
SYS_BP_STAGE2_THRESHOLD = 140.0
DIA_BP_ELEVATED_THRESHOLD = 80.0
DIA_BP_STAGE2_THRESHOLD = 90.0
CHOL_BORDERLINE_THRESHOLD = 200.0
CHOL_HIGH_THRESHOLD = 240.0
BMI_OVERWEIGHT_THRESHOLD = 25.0
BMI_OBESE_THRESHOLD = 30.0
GLUCOSE_IMPAIRED_THRESHOLD = 100.0
GLUCOSE_DIABETIC_THRESHOLD = 126.0

RISK_STYLES: dict[str, dict[str, str]] = {
    "Low": {
        "color": "#1E7E34",
        "bg": "#EAF7EE",
        "border": "#38A169",
        "badge": "LOW RISK",
    },
    "Moderate": {
        "color": "#975A16",
        "bg": "#FFFBEB",
        "border": "#D69E2E",
        "badge": "MODERATE RISK",
    },
    "High": {
        "color": "#9B2C2C",
        "bg": "#FFF5F5",
        "border": "#E53E3E",
        "badge": "HIGH RISK",
    },
}


def classify_risk_level(probability: float) -> str:
    """Map predicted CHD probability to Low (<0.10), Moderate (0.10-0.25), or High (>0.25)."""
    prob = float(probability)
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if prob < LOW_RISK_MAX:
        return "Low"
    if prob <= MODERATE_RISK_MAX:
        return "Moderate"
    return "High"


def get_risk_style(risk_level: str) -> dict[str, str]:
    """Return UI styling metadata (green/amber/red) for a given risk level."""
    if risk_level not in RISK_STYLES:
        raise ValueError(f"Unknown risk level: {risk_level}")
    return RISK_STYLES[risk_level]


def _extract_patient_dict(
    patient_data: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> dict[str, float]:
    """Normalize patient input (DataFrame, Series, or dict) into a float dict."""
    if isinstance(patient_data, pd.DataFrame):
        if len(patient_data) != 1:
            raise ValueError(f"Expected 1 patient row, got {len(patient_data)}.")
        row = patient_data.iloc[0].to_dict()
    elif isinstance(patient_data, pd.Series):
        row = patient_data.to_dict()
    else:
        row = dict(patient_data)

    cleaned: dict[str, float] = {}
    for key, val in row.items():
        if val is not None and not pd.isna(val):
            cleaned[str(key)] = float(val)
    return cleaned


def _shap_driver_map(
    top_shap_factors: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Map raw feature names to positive risk-increasing SHAP values."""
    drivers: dict[str, float] = {}
    if not top_shap_factors:
        return drivers
    for item in top_shap_factors:
        shap_val = float(item.get("shap_value", 0.0))
        if shap_val > 0:
            raw_key = str(item.get("raw_feature", item.get("feature", "")))
            drivers[raw_key] = max(drivers.get(raw_key, 0.0), shap_val)
    return drivers


def generate_recommendations(
    patient_data: pd.DataFrame | pd.Series | Mapping[str, Any],
    top_shap_factors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate personalized preventive recommendations for elevated/risky factors only.

    Evaluates smoking, blood pressure, cholesterol, BMI, and glucose against
    clinical guideline thresholds and prioritizes factors that also appear as
    top risk-increasing SHAP drivers.
    """
    p = _extract_patient_dict(patient_data)
    drivers = _shap_driver_map(top_shap_factors)
    recommendations: list[dict[str, Any]] = []

    # 1. Smoking
    current_smoker = p.get("currentSmoker", 0.0)
    cigs_per_day = p.get("cigsPerDay", 0.0)
    if current_smoker >= 1.0 or cigs_per_day >= SMOKING_CIGS_THRESHOLD:
        shap_impact = max(drivers.get("cigsPerDay", 0.0), drivers.get("currentSmoker", 0.0))
        intensity = (
            f"{int(round(cigs_per_day))} cigarettes/day"
            if cigs_per_day >= SMOKING_CIGS_THRESHOLD
            else "Active smoker"
        )
        recommendations.append(
            {
                "category": "Smoking Cessation",
                "severity": "High" if cigs_per_day >= 10 else "Elevated",
                "observed": intensity,
                "guideline_target": "0 cigarettes/day (complete cessation)",
                "is_top_shap_driver": shap_impact > 0,
                "shap_impact": round(shap_impact, 4),
                "recommendation": (
                    "Prioritize structured smoking cessation (behavioral counseling plus "
                    "nicotine replacement therapy or physician-guided pharmacotherapy). "
                    "Quitting tobacco produces one of the largest reductions in 10-year CHD risk."
                ),
            }
        )

    # 2. Blood Pressure
    sys_bp = p.get("sysBP")
    dia_bp = p.get("diaBP")
    prevalent_hyp = p.get("prevalentHyp", 0.0)
    bp_elevated = (
        (sys_bp is not None and sys_bp >= SYS_BP_ELEVATED_THRESHOLD)
        or (dia_bp is not None and dia_bp >= DIA_BP_ELEVATED_THRESHOLD)
        or prevalent_hyp >= 1.0
    )
    if bp_elevated:
        shap_impact = max(
            drivers.get("sysBP", 0.0),
            drivers.get("diaBP", 0.0),
            drivers.get("prevalentHyp", 0.0),
        )
        is_stage2 = (
            (sys_bp is not None and sys_bp >= SYS_BP_STAGE2_THRESHOLD)
            or (dia_bp is not None and dia_bp >= DIA_BP_STAGE2_THRESHOLD)
        )
        bp_text = (
            f"{sys_bp:.0f}/{dia_bp:.0f} mmHg"
            if sys_bp is not None and dia_bp is not None
            else "Hypertension history"
        )
        recommendations.append(
            {
                "category": "Blood Pressure Control",
                "severity": "High" if is_stage2 else "Elevated",
                "observed": bp_text,
                "guideline_target": "< 130 / 80 mmHg (ACC/AHA guideline)",
                "is_top_shap_driver": shap_impact > 0,
                "shap_impact": round(shap_impact, 4),
                "recommendation": (
                    "Adopt a DASH-style eating pattern (sodium < 1,500–2,300 mg/day), "
                    "perform regular aerobic activity, monitor home blood pressure, and "
                    "review antihypertensive therapy with a clinician if readings stay >= 130/80 mmHg."
                ),
            }
        )

    # 3. Total Cholesterol
    tot_chol = p.get("totChol")
    if tot_chol is not None and tot_chol >= CHOL_BORDERLINE_THRESHOLD:
        shap_impact = drivers.get("totChol", 0.0)
        is_high = tot_chol >= CHOL_HIGH_THRESHOLD
        recommendations.append(
            {
                "category": "Lipid & Cholesterol Management",
                "severity": "High" if is_high else "Borderline",
                "observed": f"{tot_chol:.0f} mg/dL",
                "guideline_target": "< 200 mg/dL desirable (< 240 mg/dL threshold)",
                "is_top_shap_driver": shap_impact > 0,
                "shap_impact": round(shap_impact, 4),
                "recommendation": (
                    "Obtain a full fasting lipid panel (LDL-C, HDL-C, triglycerides), reduce "
                    "saturated and trans fats in favor of soluble fiber and omega-3 rich foods, "
                    "and discuss statin eligibility with your physician if LDL-C remains elevated."
                    if is_high
                    else "Limit saturated fats, increase soluble fiber (oats, legumes, vegetables), "
                    "and recheck a fasting lipid panel at your next preventive visit."
                ),
            }
        )

    # 4. BMI
    bmi = p.get("BMI")
    if bmi is not None and bmi >= BMI_OVERWEIGHT_THRESHOLD:
        shap_impact = drivers.get("BMI", 0.0)
        is_obese = bmi >= BMI_OBESE_THRESHOLD
        recommendations.append(
            {
                "category": "Weight & Physical Activity (BMI)",
                "severity": "High" if is_obese else "Elevated",
                "observed": f"{bmi:.1f} kg/m²",
                "guideline_target": "18.5 – 24.9 kg/m²",
                "is_top_shap_driver": shap_impact > 0,
                "shap_impact": round(shap_impact, 4),
                "recommendation": (
                    "Aim for a gradual 5–10% weight reduction through a sustainable calorie "
                    "deficit and at least 150 minutes/week of moderate-intensity aerobic exercise "
                    "plus twice-weekly resistance training to improve blood pressure and metabolic health."
                ),
            }
        )

    # 5. Glucose / Diabetes
    glucose = p.get("glucose")
    diabetes = p.get("diabetes", 0.0)
    if (glucose is not None and glucose >= GLUCOSE_IMPAIRED_THRESHOLD) or diabetes >= 1.0:
        shap_impact = max(drivers.get("glucose", 0.0), drivers.get("diabetes", 0.0))
        is_diabetic_range = (glucose is not None and glucose >= GLUCOSE_DIABETIC_THRESHOLD) or diabetes >= 1.0
        obs_text = (
            f"Glucose {glucose:.0f} mg/dL" + (" (Diabetes history)" if diabetes >= 1.0 else "")
            if glucose is not None
            else "Diabetes history"
        )
        recommendations.append(
            {
                "category": "Glycemic & Metabolic Control",
                "severity": "High" if is_diabetic_range else "Elevated",
                "observed": obs_text,
                "guideline_target": "< 100 mg/dL fasting (ADA guideline)",
                "is_top_shap_driver": shap_impact > 0,
                "shap_impact": round(shap_impact, 4),
                "recommendation": (
                    "Schedule an HbA1c and fasting glucose evaluation with a clinician, reduce "
                    "refined carbohydrates and sugary beverages, prioritize high-fiber whole foods "
                    "and post-meal walking, and maintain prescribed diabetes therapy if diagnosed."
                ),
            }
        )

    severity_rank = {"High": 3, "Elevated": 2, "Borderline": 1}
    recommendations.sort(
        key=lambda r: (
            1 if r["is_top_shap_driver"] else 0,
            r["shap_impact"],
            severity_rank.get(r["severity"], 0),
        ),
        reverse=True,
    )
    return recommendations

