# CardioSense 🫀

> **Explainable ML System for 10-Year Coronary Heart Disease Risk Prediction**

CardioSense is an end-to-end, leakage-free machine learning pipeline built as a college mini-project. It predicts an individual's 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset, provides clinically interpretable SHAP and LIME explanations, and delivers personalised preventive recommendations through a Streamlit web dashboard.

---

## Features

- **Risk Stratification** — Low (<10%), Moderate (10–25%), High (>25%) tiers with a visual gauge
- **Leakage-Free Pipeline** — Stratified 80/20 split before imputation, scaling, or SMOTE; `ColumnTransformer` fitted only on training data
- **Class Imbalance Handling** — SMOTE applied only inside `imblearn.Pipeline` training folds
- **Model Comparison** — Logistic Regression, Random Forest, and XGBoost evaluated with stratified 5-fold cross-validation and held-out test metrics
- **Threshold Tuning** — Decision threshold optimised for CHD recall on out-of-fold predictions
- **Global SHAP** — Beeswarm summary and mean |SHAP| bar chart across the test set
- **Local SHAP & LIME** — Waterfall and LIME explanations for individual patients
- **Personalised Recommendations** — Rule-based guidance for smoking, blood pressure, cholesterol, BMI, and glucose using standard clinical thresholds
- **Streamlit Dashboard** — Sidebar input form, risk card, gauge, SHAP waterfall, factor list, recommendations, model performance tab

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Framingham Heart Study |
| Source | [Kaggle — Heart Disease Prediction](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset) |
| File | `data/framingham.csv` |
| Patients | 4,238 |
| Features | 15 (demographics, vitals, labs, medical history) |
| Target | `TenYearCHD` (binary: 1 = CHD event within 10 years) |
| Class balance | ~84.5% negative / ~15.5% positive |

---

## Tech Stack

| Layer | Libraries / Versions |
|-------|----------------------|
| Data & ML | Python 3.13, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1 |
| Imbalance | imbalanced-learn 0.14.2 |
| Boosting | xgboost 3.4.1 |
| Explainability | shap 0.54.1, lime |
| Visualisation | matplotlib 3.11.2, seaborn 0.13.2 |
| Dashboard | streamlit 1.67.0 |
| Notebooks | jupyter, ipython, nbformat |

---

## Folder Structure

```
CardioSense/
├── app/
│   └── app.py               # Streamlit dashboard
├── data/
│   └── framingham.csv       # Framingham dataset
├── models/
│   ├── best_model.pkl       # Saved best pipeline (Logistic Regression)
│   └── model_metadata.json  # Threshold, metrics, model name
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_modeling.ipynb    # Model training & evaluation
│   └── 03_explainability.ipynb  # SHAP & LIME explanations
├── reports/                 # Saved plots and CSVs
│   ├── model_comparison.csv
│   ├── roc_curves_comparison.png
│   ├── confusion_matrices.png
│   ├── threshold_tuning.png
│   ├── shap_summary_beeswarm.png
│   └── …
├── src/
│   ├── train.py             # Pipeline training & evaluation logic
│   ├── explain.py           # SHAP / LIME explanation helpers
│   └── recommend.py         # Risk stratification & recommendations
├── tests/
│   └── test_samples.py      # 5-patient end-to-end verification
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Clone the repository

```bash
git clone <repo-url>
cd CardioSense
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-run training

The pre-trained `models/best_model.pkl` is committed. To retrain from scratch:

```bash
python src/train.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Then open <http://localhost:8501> in your browser.

### 6. Run the sample patient tests

```bash
python tests/test_samples.py
```

---

## Model Results

Performance on the held-out 20% test set (stratified split, threshold tuned to 0.41):

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall | Precision | F1-Score |
|-------|------------|--------------|-------------|--------|-----------|----------|
| **Logistic Regression** ✅ | **0.7328** | **0.6985** | 0.2784 | 0.6047 | 0.2476 | 0.3514 |
| Random Forest | 0.7178 | 0.6874 | **0.2864** | 0.5426 | 0.2482 | 0.3406 |
| XGBoost | 0.7163 | 0.6730 | 0.2547 | 0.4186 | 0.2455 | 0.3095 |

> Decision threshold tuned from 0.50 → **0.41** on out-of-fold predictions to maximise CHD recall.  
> At threshold 0.41: **Tuned Recall ≈ 74.4%** (at a cost of some precision).

Top global SHAP drivers (mean |SHAP|): **Age** > Cigarettes/Day > Systolic BP > Sex (Male) > Total Cholesterol.

---

## Limitations

- **Historical cohort** — Framingham data was collected in the 1960s–80s in a predominantly white, New England population. Predictions may not generalise to contemporary or more diverse populations.
- **Class imbalance** — Only ~15.5% positive class. SMOTE improves recall but precision remains moderate.
- **Prototype only** — CardioSense is an academic project and **is not validated for clinical use**. It should not be used to make medical decisions.
- **Missing data** — Median/mode imputation is applied to missing values; this may introduce bias.
- **Feature scope** — The 15 Framingham predictors do not include modern biomarkers (e.g., CRP, LDL particle size, troponin).

---

## Screenshots

### Risk Assessment — Moderate Risk patient (10.1%)
![Risk card and gauge for a Moderate-risk patient](reports/screenshots/Screenshot%202026-11-27%20224349.png)

---

### Local SHAP Explanation — Low-risk patient
*All features push the prediction below the population baseline (blue bars).*

![SHAP waterfall and Top 5 Contributing Features for a low-risk patient](reports/screenshots/Screenshot%202026-11-27%20224356.png)

---

### Personalised Recommendations — Low-risk patient (clean bill of health)
*When no modifiable risk factors are elevated the app confirms healthy status and surfaces the medical disclaimer.*

![Recommendations panel showing no elevated risk factors and the disclaimer](reports/screenshots/Screenshot%202026-11-27%20224403.png)

---

### Risk Assessment — High Risk patient (90.3%)
*Risk card, tier badge, and gauge needle deep in the red zone.*

![Risk card and gauge for a High-risk patient](reports/screenshots/Screenshot%202026-11-27%20224722.png)

---

### Local SHAP Explanation — High-risk patient
*Age, glucose, systolic BP, smoking, and sex all push risk upward (red bars).*

![SHAP waterfall and Top 5 Contributing Features for a high-risk patient](reports/screenshots/Screenshot%202026-11-27%20224730.png)

---

### Personalised Recommendations — High-risk patient (5 actionable panels)
*Each elevated factor is expanded with an observed value, guideline target, and clinical guidance. SHAP-driver panels are labelled.*

![Five expanded recommendation panels for a high-risk patient](reports/screenshots/Screenshot%202026-11-27%20224910.png)

---

## Acknowledgements

- Framingham Heart Study — National Heart, Lung, and Blood Institute
- SHAP library — Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
- LIME library — Ribeiro et al. (2016), *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*

