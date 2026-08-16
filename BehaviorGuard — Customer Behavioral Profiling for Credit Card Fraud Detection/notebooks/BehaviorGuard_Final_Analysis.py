
"""
BehaviorGuard — Final Credit Card Fraud Detection Analysis
============================================================

Purpose
-------
This is the consolidated "final analysis" file for the project.

It is designed to finish the project after the processed behavioral dataset
has already been created. It does NOT redo the original data-cleaning/
feature-engineering work from scratch.

The script:
    1. Loads the processed behavioral dataset.
    2. Performs final data-quality checks.
    3. Creates a leakage-safe train/test split.
    4. Trains and evaluates Logistic Regression, Random Forest, and XGBoost.
    5. Reports fraud-focused metrics, confusion matrices, PR-AUC and ROC-AUC.
    6. Finds a validation-based operating threshold for XGBoost.
    7. Performs statistical analysis:
         - Mann–Whitney U
         - Benjamini–Hochberg FDR
         - Cliff's Delta
         - Chi-square
         - Cramér's V
    8. Analyses the engineered behavioral features.
    9. Produces model feature importance.
   10. Produces a compact final evidence table.
   11. Saves all important tables, plots, models and a final text summary.

Important
---------
- The target is "Class".
- "level_0" and "index" are identifiers and are excluded.
- "Transaction_Time" is not used as a raw predictor.
- Transaction hour is derived from "Time" for temporal analysis.
- Statistical tests are descriptive/inferential evidence, not model training.
- The test set is kept untouched for final model evaluation.
- The operating threshold is selected on the validation set, not the test set.

Expected processed dataset
--------------------------
../data/processed/transactions_behavioral.csv

Expected columns include:
    Time, V1...V28, Amount,
    Amount_Log, Amount_Category,
    Transactions_Last_1H, Transactions_Last_24H,
    Amount_ZScore, Class

The script is intentionally self-contained so this can serve as the final
analysis artifact for the project.
"""

# =============================================================================
# 0. IMPORTS AND CONFIGURATION
# =============================================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu, chi2_contingency, rankdata
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    balanced_accuracy_score
)

import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

DATA_PATH = Path("../data/processed/transactions_behavioral.csv")
RESULT_DIR = Path("../results/final_analysis")
FIGURE_DIR = RESULT_DIR / "figures"
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

print("=" * 80)
print("BEHAVIORGUARD — FINAL CREDIT CARD FRAUD DETECTION ANALYSIS")
print("=" * 80)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_PATH.resolve()}\n\n"
        "Make sure the processed dataset is located at "
        "../data/processed/transactions_behavioral.csv"
    )

df = pd.read_csv(DATA_PATH)

print("\n[1] DATASET")
print("-" * 80)
print("Shape:", df.shape)
print("Columns:", len(df.columns))

if "Class" not in df.columns:
    raise ValueError("Target column 'Class' was not found.")

print("\nTarget distribution:")
print(df["Class"].value_counts())

print("\nTarget percentage:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))


# =============================================================================
# 2. FINAL DATA QUALITY CHECK
# =============================================================================

print("\n[2] DATA QUALITY")
print("-" * 80)

missing = df.isna().sum()
missing = missing[missing > 0]

print("Missing values:")
print(missing if not missing.empty else "None")

print("\nDuplicate rows:", int(df.duplicated().sum()))

print("\nTarget values:", sorted(df["Class"].dropna().unique().tolist()))

if df["Class"].isna().any():
    raise ValueError("Target contains missing values.")

# Remove exact duplicate rows only if any are found.
# The project dataset previously contained zero duplicates, so this should
# normally leave the dataset unchanged.
if df.duplicated().sum() > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)


# =============================================================================
# 3. FEATURE ENGINEERING FOR FINAL ANALYSIS
# =============================================================================

print("\n[3] FEATURE SET")
print("-" * 80)

# Temporal analysis feature.
# This is derived from Time and is used for analysis, not as raw Time input.
if "Time" in df.columns:
    df["Transaction_Hour"] = (df["Time"] / 3600.0) % 24
    df["Transaction_Hour_Int"] = df["Transaction_Hour"].astype(int)

# Identifiers / technical columns that should not be learned as predictors.
EXCLUDE_COLUMNS = {
    "Class",
    "level_0",
    "index",
    "Transaction_Time",
    "Transaction_Hour",
    "Transaction_Hour_Int",
}

feature_columns = [
    c for c in df.columns
    if c not in EXCLUDE_COLUMNS
]

# Behavioral features already engineered earlier in the project.
BEHAVIORAL_FEATURES = [
    "Amount_Log",
    "Amount_Category",
    "Transactions_Last_1H",
    "Transactions_Last_24H",
    "Amount_ZScore",
]

BEHAVIORAL_FEATURES = [
    c for c in BEHAVIORAL_FEATURES if c in df.columns
]

# Numeric/categorical split.
numeric_features = [
    c for c in feature_columns
    if pd.api.types.is_numeric_dtype(df[c])
]

categorical_features = [
    c for c in feature_columns
    if not pd.api.types.is_numeric_dtype(df[c])
]

print("Total model features:", len(feature_columns))
print("Numerical features:", len(numeric_features))
print("Categorical features:", categorical_features)
print("Behavioral features:", BEHAVIORAL_FEATURES)


# =============================================================================
# 4. CLASS COUNTS
# =============================================================================

normal_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())
fraud_rate = fraud_count / len(df)

print("\n[4] CLASS BALANCE")
print("-" * 80)
print(f"Normal transactions : {normal_count:,}")
print(f"Fraud transactions  : {fraud_count:,}")
print(f"Fraud rate          : {fraud_rate:.4%}")


# =============================================================================
# 5. LEAKAGE-SAFE TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

print("\n[5] TRAIN / VALIDATION / TEST SPLIT")
print("-" * 80)

X = df[feature_columns].copy()
y = df["Class"].astype(int).copy()

# First split off 20% test data.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# From the remaining 80%, create a 20% validation split.
# This produces approximately:
#   64% train, 16% validation, 20% test.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("Train:", X_train.shape, "Fraud:", int(y_train.sum()))
print("Valid:", X_valid.shape, "Fraud:", int(y_valid.sum()))
print("Test :", X_test.shape, "Fraud:", int(y_test.sum()))


# =============================================================================
# 6. PREPROCESSING PIPELINE
# =============================================================================

# Numeric features:
#   - median imputation
#   - standardization
#
# Categorical features:
#   - most-frequent imputation
#   - one-hot encoding
#
# The preprocessing is fitted only on training data through Pipeline /
# ColumnTransformer, preventing preprocessing leakage.

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)


# =============================================================================
# 7. MODEL DEFINITIONS
# =============================================================================

# Class imbalance is severe, so class_weight="balanced" is used for the
# sklearn models. For XGBoost, scale_pos_weight is calculated from training
# data only.

scale_pos_weight = (
    (len(y_train) - int(y_train.sum()))
    / max(int(y_train.sum()), 1)
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    ),
}

# XGBoost is optional only in the sense that the script gives a clear error
# if the package is not installed. The project already used XGBoost.
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

except ImportError as exc:
    raise ImportError(
        "XGBoost is required for the final project analysis. "
        "Install it with: pip install xgboost"
    ) from exc


# =============================================================================
# 8. TRAIN MODELS AND COLLECT VALIDATION / TEST SCORES
# =============================================================================

print("\n[6] MODEL TRAINING AND EVALUATION")
print("-" * 80)

fitted_models = {}
model_results = []
validation_probabilities = {}
test_probabilities = {}

for model_name, estimator in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    fitted_models[model_name] = pipeline

    # Validation probabilities are used only for threshold selection.
    valid_prob = pipeline.predict_proba(X_valid)[:, 1]

    # Test probabilities are reserved for final evaluation.
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    validation_probabilities[model_name] = valid_prob
    test_probabilities[model_name] = test_prob

    # Default 0.50 threshold metrics.
    test_pred = (test_prob >= 0.50).astype(int)

    precision = precision_score(
        y_test, test_pred, zero_division=0
    )
    recall = recall_score(
        y_test, test_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, test_pred, zero_division=0
    )
    pr_auc = average_precision_score(
        y_test, test_prob
    )
    roc_auc = roc_auc_score(
        y_test, test_prob
    )
    balanced_acc = balanced_accuracy_score(
        y_test, test_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred
    ).ravel()

    model_results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Balanced_Accuracy": balanced_acc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

    # Save fitted model.
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / f"{safe_name}.joblib"
    )

model_results_df = pd.DataFrame(model_results)

print("\nModel benchmark:")
print(
    model_results_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1",
            "PR_AUC",
            "ROC_AUC",
            "Balanced_Accuracy",
        ]
    ].round(4).to_string(index=False)
)


# =============================================================================
# 9. CONFUSION MATRICES
# =============================================================================

print("\n[7] CONFUSION MATRICES")
print("-" * 80)

for model_name, test_prob in test_probabilities.items():

    test_pred = (test_prob >= 0.50).astype(int)
    cm = confusion_matrix(y_test, test_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )

    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =============================================================================
# 10. PRECISION-RECALL AND ROC CURVES
# =============================================================================

print("\n[8] MODEL CURVES")
print("-" * 80)

plt.figure(figsize=(8, 6))

for model_name, test_prob in test_probabilities.items():

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        test_prob
    )

    pr_auc = average_precision_score(
        y_test,
        test_prob
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{model_name} (PR-AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(8, 6))

for model_name, test_prob in test_probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        test_prob
    )

    roc_auc = roc_auc_score(
        y_test,
        test_prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (ROC-AUC={roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =============================================================================
# 11. VALIDATION-BASED THRESHOLD ANALYSIS FOR XGBOOST
# =============================================================================

print("\n[9] XGBOOST THRESHOLD ANALYSIS")
print("-" * 80)

xgb_valid_prob = validation_probabilities["XGBoost"]

threshold_grid = np.arange(
    0.05,
    0.951,
    0.01
)

threshold_rows = []

for threshold in threshold_grid:

    pred = (
        xgb_valid_prob >= threshold
    ).astype(int)

    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_valid, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, pred, zero_division=0
        ),
        "F1": f1_score(
            y_valid, pred, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_rows)

# Select threshold by validation F1.
best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_row["Threshold"]
)

print(
    "Best validation threshold by F1:",
    round(best_threshold, 2)
)

print(
    best_row.to_string()
)

threshold_df.to_csv(
    RESULT_DIR / "xgboost_threshold_analysis.csv",
    index=False
)


# Apply the selected threshold once to the untouched test set.
xgb_test_prob = test_probabilities["XGBoost"]

xgb_default_pred = (
    xgb_test_prob >= 0.50
).astype(int)

xgb_tuned_pred = (
    xgb_test_prob >= best_threshold
).astype(int)

threshold_comparison = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1"
    ],

    "Default_0.50": [
        precision_score(
            y_test,
            xgb_default_pred,
            zero_division=0
        ),
        recall_score(
            y_test,
            xgb_default_pred,
            zero_division=0
        ),
        f1_score(
            y_test,
            xgb_default_pred,
            zero_division=0
        )
    ],

    "Validation_Selected_Threshold": [
        precision_score(
            y_test,
            xgb_tuned_pred,
            zero_division=0
        ),
        recall_score(
            y_test,
            xgb_tuned_pred,
            zero_division=0
        ),
        f1_score(
            y_test,
            xgb_tuned_pred,
            zero_division=0
        )
    ]
})

threshold_comparison.to_csv(
    RESULT_DIR / "xgboost_threshold_comparison.csv",
    index=False
)

print("\nXGBoost threshold comparison:")
print(
    threshold_comparison.round(4).to_string(index=False)
)


# =============================================================================
# 12. STATISTICAL ANALYSIS — NUMERICAL FEATURES
# =============================================================================

print("\n[10] STATISTICAL ANALYSIS")
print("-" * 80)

normal_df = df[df["Class"] == 0]
fraud_df = df[df["Class"] == 1]

statistical_numeric_features = [
    c for c in df.columns
    if c not in {
        "Class",
        "level_0",
        "index",
        "Transaction_Time",
        "Transaction_Hour_Int",
    }
    and pd.api.types.is_numeric_dtype(df[c])
]


def cliffs_delta(x, y):
    """
    Cliff's Delta for two independent samples.

    x = fraud group
    y = normal group

    Positive:
        fraud values tend to be larger.

    Negative:
        fraud values tend to be smaller.

    Magnitude:
        < 0.147   negligible
        < 0.33    small
        < 0.474   medium
        >= 0.474  large
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx = len(x)
    ny = len(y)

    if nx == 0 or ny == 0:
        return np.nan

    combined = np.concatenate([x, y])
    ranks = rankdata(combined)

    rank_x = ranks[:nx]

    U = (
        rank_x.sum()
        - nx * (nx + 1) / 2
    )

    return (
        (2 * U) / (nx * ny)
    ) - 1


def effect_label(delta):
    magnitude = abs(delta)

    if magnitude < 0.147:
        return "Negligible"
    elif magnitude < 0.33:
        return "Small"
    elif magnitude < 0.474:
        return "Medium"
    return "Large"


stat_rows = []

for feature in statistical_numeric_features:

    normal_values = normal_df[feature].dropna().values
    fraud_values = fraud_df[feature].dropna().values

    if len(normal_values) == 0 or len(fraud_values) == 0:
        continue

    u_stat, p_value = mannwhitneyu(
        normal_values,
        fraud_values,
        alternative="two-sided"
    )

    delta = cliffs_delta(
        fraud_values,
        normal_values
    )

    stat_rows.append({
        "Feature": feature,
        "Normal_Median": np.median(normal_values),
        "Fraud_Median": np.median(fraud_values),
        "U_Statistic": u_stat,
        "P_Value": p_value,
        "Cliffs_Delta": delta,
        "Absolute_Cliffs_Delta": abs(delta),
        "Effect_Size": effect_label(delta),
    })

stat_df = pd.DataFrame(stat_rows)

reject, adjusted_p, _, _ = multipletests(
    stat_df["P_Value"],
    alpha=0.05,
    method="fdr_bh"
)

stat_df["Adjusted_P_Value"] = adjusted_p
stat_df["Significant"] = reject

stat_df = stat_df.sort_values(
    "Absolute_Cliffs_Delta",
    ascending=False
)

stat_df.to_csv(
    RESULT_DIR / "statistical_numerical_features.csv",
    index=False
)

print("\nTop statistical features:")
print(
    stat_df[
        [
            "Feature",
            "Normal_Median",
            "Fraud_Median",
            "Adjusted_P_Value",
            "Cliffs_Delta",
            "Effect_Size",
        ]
    ].head(20).round(6).to_string(index=False)
)


# =============================================================================
# 13. CATEGORICAL ANALYSIS
# =============================================================================

print("\n[11] CATEGORICAL ANALYSIS")
print("-" * 80)

categorical_results = []

for feature in [
    c for c in df.columns
    if c not in {"Class", "level_0", "index", "Transaction_Time"}
    and not pd.api.types.is_numeric_dtype(df[c])
]:

    table = pd.crosstab(
        df[feature],
        df["Class"]
    )

    if table.shape[0] < 2 or table.shape[1] < 2:
        continue

    chi2, p_value, dof, expected = chi2_contingency(
        table
    )

    n = table.to_numpy().sum()
    rows, cols = table.shape

    phi2 = chi2 / n

    # Bias-corrected Cramér's V.
    phi2_corrected = max(
        0,
        phi2
        - ((cols - 1) * (rows - 1)) / max(n - 1, 1)
    )

    rows_corrected = (
        rows
        - ((rows - 1) ** 2) / max(n - 1, 1)
    )

    cols_corrected = (
        cols
        - ((cols - 1) ** 2) / max(n - 1, 1)
    )

    denominator = min(
        cols_corrected - 1,
        rows_corrected - 1
    )

    cramers_v = (
        np.sqrt(phi2_corrected / denominator)
        if denominator > 0
        else np.nan
    )

    categorical_results.append({
        "Feature": feature,
        "Chi2": chi2,
        "P_Value": p_value,
        "Degrees_of_Freedom": dof,
        "Cramers_V": cramers_v,
    })

categorical_df = pd.DataFrame(
    categorical_results
)

if not categorical_df.empty:

    categorical_df["Adjusted_P_Value"] = (
        multipletests(
            categorical_df["P_Value"],
            alpha=0.05,
            method="fdr_bh"
        )[1]
    )

    categorical_df.to_csv(
        RESULT_DIR / "statistical_categorical_features.csv",
        index=False
    )

    print(
        categorical_df.round(6).to_string(index=False)
    )


# =============================================================================
# 14. BEHAVIORAL FEATURE REPORT
# =============================================================================

print("\n[12] BEHAVIORAL FEATURE EVIDENCE")
print("-" * 80)

behavioral_numeric = [
    c for c in BEHAVIORAL_FEATURES
    if c in stat_df["Feature"].values
]

behavioral_stat_df = stat_df[
    stat_df["Feature"].isin(behavioral_numeric)
].copy()

print(
    behavioral_stat_df[
        [
            "Feature",
            "Normal_Median",
            "Fraud_Median",
            "Adjusted_P_Value",
            "Significant",
            "Cliffs_Delta",
            "Effect_Size",
        ]
    ].round(6).to_string(index=False)
)

behavioral_stat_df.to_csv(
    RESULT_DIR / "behavioral_feature_evidence.csv",
    index=False
)


# =============================================================================
# 15. BEHAVIORAL FEATURE CORRELATION
# =============================================================================

behavioral_numeric_for_corr = [
    c for c in BEHAVIORAL_FEATURES
    if c in df.columns
    and pd.api.types.is_numeric_dtype(df[c])
]

if len(behavioral_numeric_for_corr) >= 2:

    behavioral_corr = df[
        behavioral_numeric_for_corr
    ].corr()

    behavioral_corr.to_csv(
        RESULT_DIR / "behavioral_feature_correlation.csv"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        behavioral_corr,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="coolwarm"
    )

    plt.title("Behavioral Feature Correlation")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "behavioral_feature_correlation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 16. FRAUD RATE BY AMOUNT CATEGORY
# =============================================================================

if "Amount_Category" in df.columns:

    amount_category = (
        df.groupby(
            "Amount_Category",
            observed=True
        )
        .agg(
            Transactions=("Class", "count"),
            Fraud_Count=("Class", "sum")
        )
        .reset_index()
    )

    amount_category["Fraud_Rate_Percent"] = (
        amount_category["Fraud_Count"]
        / amount_category["Transactions"]
        * 100
    )

    amount_category.to_csv(
        RESULT_DIR / "amount_category_fraud_rate.csv",
        index=False
    )

    table = pd.crosstab(
        df["Amount_Category"],
        df["Class"]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        table
    )

    n = table.to_numpy().sum()
    rows, cols = table.shape

    phi2 = chi2 / n

    phi2_corrected = max(
        0,
        phi2
        - ((cols - 1) * (rows - 1)) / max(n - 1, 1)
    )

    rows_corrected = (
        rows
        - ((rows - 1) ** 2) / max(n - 1, 1)
    )

    cols_corrected = (
        cols
        - ((cols - 1) ** 2) / max(n - 1, 1)
    )

    denominator = min(
        cols_corrected - 1,
        rows_corrected - 1
    )

    amount_cramers_v = (
        np.sqrt(phi2_corrected / denominator)
        if denominator > 0
        else np.nan
    )

    amount_test = pd.DataFrame({
        "Test": [
            "Chi-square",
            "Cramer's V"
        ],
        "Value": [
            chi2,
            amount_cramers_v
        ],
        "P_Value": [
            p_value,
            np.nan
        ]
    })

    amount_test.to_csv(
        RESULT_DIR / "amount_category_test.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=amount_category,
        x="Amount_Category",
        y="Fraud_Rate_Percent"
    )

    plt.title("Fraud Rate by Amount Category")
    plt.xlabel("Amount Category")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "amount_category_fraud_rate.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nAmount Category:")
    print(f"Chi-square : {chi2:.6f}")
    print(f"p-value    : {p_value:.6e}")
    print(f"Cramer's V : {amount_cramers_v:.6f}")


# =============================================================================
# 17. TEMPORAL FRAUD ANALYSIS
# =============================================================================

if "Transaction_Hour_Int" in df.columns:

    hourly = (
        df.groupby("Transaction_Hour_Int")
        .agg(
            Transactions=("Class", "count"),
            Fraud_Count=("Class", "sum")
        )
        .reset_index()
    )

    hourly["Fraud_Rate_Percent"] = (
        hourly["Fraud_Count"]
        / hourly["Transactions"]
        * 100
    )

    hourly.to_csv(
        RESULT_DIR / "hourly_fraud_analysis.csv",
        index=False
    )

    plt.figure(figsize=(11, 5))

    sns.lineplot(
        data=hourly,
        x="Transaction_Hour_Int",
        y="Fraud_Rate_Percent",
        marker="o"
    )

    plt.xticks(range(24))
    plt.xlabel("Hour of Day")
    plt.ylabel("Fraud Rate (%)")
    plt.title("Fraud Rate by Transaction Hour")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "fraud_rate_by_hour.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 18. XGBOOST FEATURE IMPORTANCE
# =============================================================================

print("\n[13] XGBOOST FEATURE IMPORTANCE")
print("-" * 80)

xgb_pipeline = fitted_models["XGBoost"]

preprocessor_fitted = xgb_pipeline.named_steps["preprocessor"]
xgb_model = xgb_pipeline.named_steps["model"]

try:
    transformed_names = (
        preprocessor_fitted
        .get_feature_names_out()
    )

    importance_values = (
        xgb_model.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": transformed_names,
        "Importance": importance_values
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        RESULT_DIR / "xgboost_feature_importance.csv",
        index=False
    )

    top_importance = importance_df.head(20)

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "xgboost_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        top_importance.round(6).to_string(index=False)
    )

except Exception as exc:
    print(
        "Feature importance extraction skipped:",
        repr(exc)
    )
    importance_df = pd.DataFrame()


# =============================================================================
# 19. MODEL VS STATISTICAL EVIDENCE
# =============================================================================

# This table does not claim that statistical significance equals predictive
# importance. It simply puts the evidence sources side-by-side.

stat_lookup = stat_df.set_index("Feature")

comparison_rows = []

for feature in BEHAVIORAL_FEATURES:

    if feature not in stat_lookup.index:
        continue

    row = stat_lookup.loc[feature]

    comparison_rows.append({
        "Feature": feature,
        "Adjusted_P_Value": row["Adjusted_P_Value"],
        "Cliffs_Delta": row["Cliffs_Delta"],
        "Effect_Size": row["Effect_Size"],
    })

behavioral_evidence_df = pd.DataFrame(
    comparison_rows
)

behavioral_evidence_df.to_csv(
    RESULT_DIR / "behavioral_evidence_summary.csv",
    index=False
)


# =============================================================================
# 20. SAVE MODEL BENCHMARK
# =============================================================================

model_results_df.to_csv(
    RESULT_DIR / "model_benchmark.csv",
    index=False
)

threshold_comparison.to_csv(
    RESULT_DIR / "xgboost_threshold_comparison.csv",
    index=False
)


# =============================================================================
# 21. FINAL PROJECT SUMMARY
# =============================================================================

best_model_row = model_results_df.loc[
    model_results_df["PR_AUC"].idxmax()
]

best_model_name = best_model_row["Model"]

# Number of statistically strong numerical features.
large_effect_count = int(
    (stat_df["Effect_Size"] == "Large").sum()
)

medium_effect_count = int(
    (stat_df["Effect_Size"] == "Medium").sum()
)

small_effect_count = int(
    (stat_df["Effect_Size"] == "Small").sum()
)

summary = {
    "dataset_size": int(len(df)),
    "normal_transactions": normal_count,
    "fraud_transactions": fraud_count,
    "fraud_rate": float(fraud_rate),

    "best_model_by_pr_auc": best_model_name,
    "best_model_pr_auc": float(best_model_row["PR_AUC"]),
    "best_model_roc_auc": float(best_model_row["ROC_AUC"]),
    "best_model_precision": float(best_model_row["Precision"]),
    "best_model_recall": float(best_model_row["Recall"]),
    "best_model_f1": float(best_model_row["F1"]),

    "xgboost_validation_best_threshold": best_threshold,

    "large_effect_numerical_features": large_effect_count,
    "medium_effect_numerical_features": medium_effect_count,
    "small_effect_numerical_features": small_effect_count,

    "behavioral_features_analyzed": BEHAVIORAL_FEATURES,
}

with open(
    RESULT_DIR / "final_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )


# Human-readable summary.
with open(
    RESULT_DIR / "FINAL_PROJECT_SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BEHAVIORGUARD — FINAL PROJECT SUMMARY\n"
        "=====================================\n\n"
    )

    f.write(
        f"Dataset size: {len(df):,}\n"
    )

    f.write(
        f"Normal transactions: {normal_count:,}\n"
    )

    f.write(
        f"Fraud transactions: {fraud_count:,}\n"
    )

    f.write(
        f"Fraud rate: {fraud_rate:.4%}\n\n"
    )

    f.write(
        "MODEL BENCHMARK\n"
        "---------------\n"
    )

    f.write(
        model_results_df[
            [
                "Model",
                "Precision",
                "Recall",
                "F1",
                "PR_AUC",
                "ROC_AUC",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        f"Best model by PR-AUC: {best_model_name}\n"
    )

    f.write(
        f"XGBoost validation-selected threshold: "
        f"{best_threshold:.2f}\n\n"
    )

    f.write(
        "TOP STATISTICAL FEATURES\n"
        "------------------------\n"
    )

    f.write(
        stat_df[
            [
                "Feature",
                "Adjusted_P_Value",
                "Cliffs_Delta",
                "Effect_Size",
            ]
        ]
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "BEHAVIORAL FEATURES\n"
        "-------------------\n"
    )

    if not behavioral_evidence_df.empty:

        f.write(
            behavioral_evidence_df.round(6)
            .to_string(index=False)
        )

    f.write("\n\n")

    f.write(
        "INTERPRETATION\n"
        "--------------\n"
        "1. The dataset is extremely imbalanced, so PR-AUC, recall, "
        "precision and F1 are more informative than accuracy alone.\n"
        "2. Several anonymized V-features show very strong statistical "
        "separation between normal and fraudulent transactions.\n"
        "3. Engineered behavioral features should be interpreted as "
        "complementary signals rather than automatically assuming they "
        "are individually strong fraud predictors.\n"
        "4. Statistical significance is interpreted together with effect "
        "size because the dataset is large.\n"
        "5. The XGBoost threshold is selected on validation data and the "
        "test set is used only for final evaluation.\n"
    )


# =============================================================================
# 22. FINAL CONSOLE REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL PROJECT REPORT")
print("=" * 80)

print(f"\nDataset size       : {len(df):,}")
print(f"Normal transactions: {normal_count:,}")
print(f"Fraud transactions : {fraud_count:,}")
print(f"Fraud rate         : {fraud_rate:.4%}")

print("\nModel benchmark:")
print(
    model_results_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1",
            "PR_AUC",
            "ROC_AUC",
        ]
    ].round(4).to_string(index=False)
)

print(
    f"\nBest model by PR-AUC: {best_model_name}"
)

print(
    f"XGBoost validation-selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"\nLarge-effect numerical features: {large_effect_count}"
)

print(
    f"Medium-effect numerical features: {medium_effect_count}"
)

print(
    f"Small-effect numerical features: {small_effect_count}"
)

print("\nTop 10 statistical features:")
print(
    stat_df[
        [
            "Feature",
            "Adjusted_P_Value",
            "Cliffs_Delta",
            "Effect_Size",
        ]
    ]
    .head(10)
    .round(6)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nAll final outputs were saved to:"
    f"\n{RESULT_DIR.resolve()}"
)

print(
    "\nImportant files:"
    "\n- model_benchmark.csv"
    "\n- xgboost_threshold_comparison.csv"
    "\n- statistical_numerical_features.csv"
    "\n- statistical_categorical_features.csv"
    "\n- behavioral_feature_evidence.csv"
    "\n- xgboost_feature_importance.csv"
    "\n- FINAL_PROJECT_SUMMARY.txt"
    "\n- final_summary.json"
    "\n- figures/"
    "\n- models/"
)
