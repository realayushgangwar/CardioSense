"""Training, tuning, evaluation, and threshold optimization for CardioSense.

All preprocessing (imputation, scaling) and resampling (SMOTE) live inside an
``imblearn.pipeline.Pipeline`` and are executed strictly after an 80/20
stratified train-test split to prevent any data or label leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COL = "TenYearCHD"

NUMERIC_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

CATEGORICAL_FEATURES = [
    "male",
    "education",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]


def load_and_split_data(
    data_path: Path,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the Framingham dataset and perform a stratified 80/20 split.

    LEAKAGE PREVENTION: Splitting happens immediately after loading, before any
    imputation, scaling, SMOTE resampling, or threshold tuning. Holding out
    ``X_test`` and ``y_test`` up front ensures test statistics never influence
    medians, modes, scaler mean/variance, or synthetic SMOTE neighbors.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Create the feature preprocessing transformer.

    LEAKAGE PREVENTION: Returning an *unfitted* ``ColumnTransformer`` lets us
    embed it inside the CV/training pipeline so medians, most-frequent modes,
    and StandardScaler parameters are computed strictly on each training fold.
    """
    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_model_search_spaces(random_state: int = RANDOM_STATE) -> dict[str, dict[str, Any]]:
    """Define base estimators and compact RandomizedSearchCV parameter spaces."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                solver="saga",
                max_iter=3000,
                random_state=random_state,
            ),
            "params": {
                "classifier__C": np.logspace(-3, 2, 20),
                "classifier__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                "smote__k_neighbors": [3, 5, 7],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            "params": {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [4, 6, 8, 12, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [2, 4, 8, 12],
                "classifier__max_features": ["sqrt", "log2", 0.5],
                "smote__k_neighbors": [3, 5, 7],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            "params": {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [2, 3, 4, 5],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.85, 1.0],
                "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                "classifier__min_child_weight": [1, 3, 5, 10],
                "classifier__reg_lambda": [1.0, 5.0, 10.0],
                "smote__k_neighbors": [3, 5, 7],
            },
        },
    }


def build_training_pipeline(classifier: Any, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Wrap preprocessing, SMOTE, and classifier inside an imblearn Pipeline.

    LEAKAGE PREVENTION: Using ``imblearn.pipeline.Pipeline`` ensures ``SMOTE``
    runs ONLY during ``fit`` (on the training folds inside CV) and is skipped
    automatically during ``predict`` / ``predict_proba`` on validation folds
    and the held-out test set.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", classifier),
        ]
    )


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    n_iter: int = 15,
    random_state: int = RANDOM_STATE,
) -> dict[str, RandomizedSearchCV]:
    """Run stratified 5-fold RandomizedSearchCV for all candidate models."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search_spaces = get_model_search_spaces(random_state=random_state)
    searches: dict[str, RandomizedSearchCV] = {}

    for name, cfg in search_spaces.items():
        pipeline = build_training_pipeline(cfg["estimator"], random_state=random_state)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=cfg["params"],
            n_iter=n_iter,
            scoring={
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
                "recall": "recall",
                "f1": "f1",
            },
            refit="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search

    return searches


def evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a fitted pipeline on the held-out test set at a given threshold."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "model": model_name,
        "threshold": round(float(threshold), 3),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_proba": y_proba,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_all_models(
    searches: dict[str, RandomizedSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate all tuned pipelines on X_test and build a comparison DataFrame."""
    details: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for name, search in searches.items():
        res = evaluate_model(name, search.best_estimator_, X_test, y_test, threshold=threshold)
        res["cv_best_roc_auc"] = float(search.best_score_)
        details[name] = res
        rows.append(
            {
                "Model": name,
                "CV ROC-AUC": round(res["cv_best_roc_auc"], 4),
                "Test ROC-AUC": round(res["roc_auc"], 4),
                "Test PR-AUC": round(res["pr_auc"], 4),
                "Recall": round(res["recall"], 4),
                "Precision": round(res["precision"], 4),
                "F1-Score": round(res["f1"], 4),
            }
        )

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("Test ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    return comparison_df, details


def tune_threshold_cv(
    best_pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    beta: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    """Select an optimal probability threshold using out-of-fold CV on X_train.

    LEAKAGE PREVENTION: Threshold selection uses ``cross_val_predict`` strictly
    on ``X_train`` (where each fold fits preprocessing + SMOTE only on its 4
    training folds). Never optimizing the threshold on ``X_test`` keeps the
    test evaluation an unbiased estimate of generalization performance.

    Maximizes F-beta (default ``beta=2.0``) to emphasize CHD recall while
    penalizing extreme precision collapse.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.round(np.arange(0.15, 0.81, 0.01), 2)
    records = []
    for t in thresholds:
        preds = (oof_proba >= t).astype(int)
        records.append(
            {
                "threshold": float(t),
                "recall": float(recall_score(y_train, preds, zero_division=0)),
                "precision": float(precision_score(y_train, preds, zero_division=0)),
                "f1": float(f1_score(y_train, preds, zero_division=0)),
                "f2": float(fbeta_score(y_train, preds, beta=beta, zero_division=0)),
            }
        )

    sweep_df = pd.DataFrame(records)
    best_idx = int(sweep_df["f2"].idxmax())
    best_threshold = float(sweep_df.loc[best_idx, "threshold"])
    return best_threshold, sweep_df


def plot_roc_curves(
    eval_details: dict[str, dict[str, Any]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for all candidate models on the test set."""
    colors = {
        "Logistic Regression": "#2B6CB0",
        "Random Forest": "#2F855A",
        "XGBoost": "#D94848",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, res in eval_details.items():
        ax.plot(
            res["fpr"],
            res["tpr"],
            linewidth=2.2,
            color=colors.get(name),
            label=f"{name} (ROC-AUC = {res['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#718096", linewidth=1.3, label="Random Chance (0.50)")
    ax.set_title("Test Set ROC Curves Comparison (Threshold-Independent)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    eval_details: dict[str, dict[str, Any]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot side-by-side test confusion matrices at the default 0.50 threshold."""
    n_models = len(eval_details)
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 4.4))
    if n_models == 1:
        axes = [axes]

    labels = ["No CHD (0)", "CHD (1)"]
    for ax, (name, res) in zip(axes, eval_details.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.8,
            ax=ax,
        )
        ax.set_title(f"{name}\n(Recall={res['recall']:.2f}, Prec={res['precision']:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle("Test Set Confusion Matrices (Default Threshold = 0.50)", fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_threshold_tradeoff(
    sweep_df: pd.DataFrame,
    best_threshold: float,
    best_model_name: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot Recall, Precision, F1, and F2 across thresholds on CV out-of-fold probabilities."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall (CHD Sensitivity)", color="#D94848", linewidth=2.2)
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision (PPV)", color="#2B6CB0", linewidth=2.2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1-Score", color="#2F855A", linewidth=2.0)
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2-Score (Recall-weighted)", color="#805AD5", linewidth=2.0, linestyle="-.")

    ax.axvline(0.50, color="#718096", linestyle="--", linewidth=1.4, label="Default Threshold (0.50)")
    ax.axvline(best_threshold, color="#1A202C", linestyle=":", linewidth=2.0, label=f"Tuned Threshold ({best_threshold:.2f})")

    ax.set_title(f"Decision Threshold Tuning on Training OOF Predictions ({best_model_name})", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (Predicted Probability >= t -> CHD)")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def save_artifacts(
    best_model_name: str,
    best_pipeline: ImbPipeline,
    best_threshold: float,
    comparison_df: pd.DataFrame,
    tuned_test_metrics: dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
) -> None:
    """Persist the winning pipeline, threshold metadata, and comparison CSV."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)

    metadata_path = models_dir / "model_metadata.json"
    metadata = {
        "best_model": best_model_name,
        "default_threshold": 0.50,
        "tuned_threshold": round(float(best_threshold), 3),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_metrics_at_tuned_threshold": {
            "recall": round(tuned_test_metrics["recall"], 4),
            "precision": round(tuned_test_metrics["precision"], 4),
            "f1": round(tuned_test_metrics["f1"], 4),
            "roc_auc": round(tuned_test_metrics["roc_auc"], 4),
            "pr_auc": round(tuned_test_metrics["pr_auc"], 4),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)


def main() -> None:
    """End-to-end leakage-free training entrypoint."""
    import matplotlib
    matplotlib.use("Agg")

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "framingham.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading dataset and performing stratified 80/20 split...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    print(f"   Train: {X_train.shape[0]} rows (CHD rate={y_train.mean():.4f})")
    print(f"   Test:  {X_test.shape[0]} rows (CHD rate={y_test.mean():.4f})")

    print("2. Training & tuning Logistic Regression, Random Forest, and XGBoost (5-fold CV + SMOTE)...")
    searches = train_and_tune_models(X_train, y_train)

    print("3. Evaluating models on held-out test set (threshold=0.50)...")
    comparison_df, eval_details = evaluate_all_models(searches, X_test, y_test, threshold=0.50)
    print(comparison_df.to_string(index=False))

    best_model_name = str(comparison_df.iloc[0]["Model"])
    best_pipeline = searches[best_model_name].best_estimator_

    print(f"4. Tuning decision threshold on training CV out-of-fold probabilities for {best_model_name}...")
    best_threshold, sweep_df = tune_threshold_cv(best_pipeline, X_train, y_train)
    tuned_test_metrics = evaluate_model(
        best_model_name, best_pipeline, X_test, y_test, threshold=best_threshold
    )
    print(
        f"   Tuned threshold={best_threshold:.2f} -> Test Recall={tuned_test_metrics['recall']:.4f}, "
        f"Precision={tuned_test_metrics['precision']:.4f}, F1={tuned_test_metrics['f1']:.4f}"
    )

    print("5. Saving plots, comparison table, and best_model.pkl...")
    fig_roc = plot_roc_curves(eval_details, save_path=reports_dir / "roc_curves_comparison.png")
    plt.close(fig_roc)
    fig_cm = plot_confusion_matrices(eval_details, save_path=reports_dir / "confusion_matrices.png")
    plt.close(fig_cm)
    fig_thr = plot_threshold_tradeoff(
        sweep_df,
        best_threshold,
        best_model_name,
        save_path=reports_dir / "threshold_tuning.png",
    )
    plt.close(fig_thr)

    save_artifacts(
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        best_threshold=best_threshold,
        comparison_df=comparison_df,
        tuned_test_metrics=tuned_test_metrics,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )
    print("Done. Artifacts saved to models/ and reports/.")


if __name__ == "__main__":
    main()

