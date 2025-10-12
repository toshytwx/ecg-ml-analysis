#!/usr/bin/env python3
"""
ecg_age_group_logreg.py

Single-file pipeline:
- read CSV
- filter features (drop many-NaN cols and extremely small-magnitude cols)
- select top-K features (SelectKBest) with K >= 10 (here K=12)
- train/test split, SMOTE on train
- Robust scaling
- GridSearchCV for LogisticRegression
- diagnostics and plots:
    - class distribution before/after SMOTE
    - correlation heatmap for selected features
    - boxplots of a few top features by age_group
    - learning curve (train vs validation)
    - CV results plot
    - confusion matrix
    - feature importance (mean abs coef)
- saves plots to files and shows them
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")  # keep output tidy; remove if debugging

# ---------------- CONFIG ----------------
CSV_FILE = "./output/ecg_features.csv"   # change if needed
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "./ecg_linear_outputs"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "all_plots.png")
K_FEATURES = 20   # choose at least 10 features (user requested min 10)
SMOTE_K_NEIGHBORS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("Loading:", CSV_FILE)
data = pd.read_csv(CSV_FILE)
print("Shape:", data.shape)

# If 'record' column exists but isn't needed, keep it aside (not used as feature)
if "record" in data.columns:
    record_col = data["record"]
else:
    record_col = None

# ---------------- PREPROCESS MISSING VALUES ----------------
# Fill missing values with the least common (mode-like but least frequent) within column as in baseline
from collections import Counter as _Counter
for col in data.columns:
    if data[col].isnull().any():
        counts = _Counter(data[col].dropna())
        if counts:
            least_common_value = counts.most_common()[-1][0]
            data[col].fillna(least_common_value, inplace=True)

# ---------------- FEATURE FILTERING ----------------
# Remove explicit non-feature columns if present
drop_cols = {"record", "age_group", "gender", "device"}
available_cols = [c for c in data.columns if c not in drop_cols]

# Drop columns with more than 30% missing values (fallback; should be none after fill, but keep for safety)
threshold = 0.3 * len(data)
valid_features = [c for c in available_cols if data[c].count() >= threshold]

# Remove features with extremely small magnitude (mean absolute < 1e-6)
filtered_features = [c for c in valid_features if np.nanmean(np.abs(data[c])) > 1e-6]

# Remove zero-variance features
filtered_features = [c for c in filtered_features if data[c].nunique() > 1]

print(f"Candidate features after filtering: {len(filtered_features)}")
print(filtered_features)

# ---------------- TARGET ----------------
if "age_group" not in data.columns:
    raise RuntimeError("CSV must contain 'age_group' column")
y = data["age_group"].astype(int)

# ---------------- SELECT TOP-K FEATURES ----------------
X_all = data[filtered_features].copy()

# If we have fewer than K_FEATURES available, reduce K accordingly
K = min(K_FEATURES, X_all.shape[1])
if K < 10:
    print("Warning: fewer than 10 valid features available, using all available features:", X_all.shape[1])

# Use SelectKBest (ANOVA F-test) to choose features predictive of multiclass target
selector = SelectKBest(score_func=f_classif, k=K)
# Fill any leftover NaNs (shouldn't exist) with column median
X_all = X_all.fillna(X_all.median())
selector.fit(X_all, y)
selected_mask = selector.get_support()
selected_features = X_all.columns[selected_mask].tolist()

print(f"Selected top-{K} features ({len(selected_features)}):")
print(selected_features)

X = X_all[selected_features].copy()

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Class distribution (train):", np.bincount(y_train))
print("Class distribution (test):", np.bincount(y_test))

# ---------------- HANDLE CLASS IMBALANCE (SMOTE) ----------------
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(SMOTE_K_NEIGHBORS, max(1, int(np.min(np.bincount(y_train)) - 1))))
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Class distribution before SMOTE:", np.bincount(y_train))
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# ---------------- FEATURE SCALING ----------------
scaler = RobustScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Remove any columns that became constant or inf/nan after scaling
def clean_numeric_arrays(X_arr, feature_names):
    # X_arr: numpy array
    # feature_names: list
    mask_ok = np.isfinite(X_arr).all(axis=0) & (np.nanstd(X_arr, axis=0) > 0)
    if not mask_ok.all():
        keep = [f for f, ok in zip(feature_names, mask_ok) if ok]
        X_arr = X_arr[:, mask_ok]
        print("Dropped features due to NaN/Inf/zero-variance after scaling:", [f for f in feature_names if f not in keep])
        feature_names = keep
    return X_arr, feature_names

X_train_res_scaled, selected_features = clean_numeric_arrays(X_train_res_scaled, selected_features)
X_test_scaled, selected_features = clean_numeric_arrays(X_test_scaled, selected_features)

# ---------------- GRID SEARCH FOR LOGISTIC REGRESSION ----------------
# Build a logistic regression that's suitable for multiclass:
# allow both 'liblinear' (ovr) and 'saga' (multinomial option) in grid
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "class_weight": ["balanced"]
}

log_reg = LogisticRegression(max_iter=20000, random_state=RANDOM_STATE)
grid = GridSearchCV(log_reg, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train_res_scaled, y_train_res)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_

# ---------------- EVALUATION ----------------
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

# ---------------- PLOTS ----------------
sns.set(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# 1) Class distribution before / after SMOTE
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4))
train_counts = Counter(y_train)
res_counts = Counter(y_train_res)
ax1[0].bar(list(train_counts.keys()), list(train_counts.values()))
ax1[0].set_title("Train class distribution (before SMOTE)")
ax1[0].set_xlabel("age_group")
ax1[0].set_ylabel("count")

ax1[1].bar(list(res_counts.keys()), list(res_counts.values()))
ax1[1].set_title("Train class distribution (after SMOTE)")
ax1[1].set_xlabel("age_group")
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))

# 2) Correlation heatmap for selected features
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
corr = pd.DataFrame(X_train_res, columns=selected_features).corr()
sns.heatmap(corr, annot=True, fmt=".2f", square=True, ax=ax2)
ax2.set_title("Feature correlation (train, selected features)")
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"))

# 3) Boxplots/violinplots of top 4 features by f-score (SelectKBest scores)
# compute column scores from selector (we have selector.scores_ that corresponds to X_all.columns)
# build DataFrame with selected features and y (original)
df_for_plots = pd.concat([X[selected_features].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
top4 = selected_features[:4]  # good heuristic
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
axes3 = axes3.flatten()
for ax, col in zip(axes3, top4):
    sns.boxplot(x="age_group", y=col, data=df_for_plots, ax=ax)
    ax.set_title(f"{col} by age_group")
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "boxplots_top4.png"))

# 4) Learning curve (train vs cross-val)
train_sizes, train_scores, test_scores = learning_curve(
    best_model, np.vstack([X_train_res_scaled, X_test_scaled])[:len(X_train_res_scaled)], np.concatenate([y_train_res, y_test])[:len(y_train_res)],
    cv=3, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=RANDOM_STATE
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(train_sizes, train_scores_mean, marker="o", label="Training score")
ax4.plot(train_sizes, test_scores_mean, marker="o", label="Cross-validation score")
ax4.set_xlabel("Number of training examples")
ax4.set_ylabel("Score (accuracy)")
ax4.set_title("Learning Curve")
ax4.legend()
ax4.grid(True)
fig4.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))

# 5) GridSearch CV results: mean_test_score vs C, separated by penalty
results = pd.DataFrame(grid.cv_results_)
fig5, ax5 = plt.subplots(figsize=(8, 6))
for penalty in results["param_penalty"].unique():
    subset = results[results["param_penalty"] == penalty]
    # some solvers may be incompatible and produce NaN; drop NaNs
    subset = subset.dropna(subset=["param_C", "mean_test_score"])
    ax5.plot(subset["param_C"].astype(float), subset["mean_test_score"], marker="o", label=f"penalty={penalty}")
ax5.set_xscale("log")
ax5.set_xlabel("C (regularization inverse strength)")
ax5.set_ylabel("Mean CV accuracy")
ax5.set_title("GridSearchCV results by penalty")
ax5.legend()
ax5.grid(True)
fig5.savefig(os.path.join(OUTPUT_DIR, "gridsearch_results.png"))

# 6) Confusion matrix heatmap
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax6)
ax6.set_xlabel("Predicted age_group")
ax6.set_ylabel("True age_group")
ax6.set_title(f"Confusion Matrix (Test Accuracy: {acc:.3f})")
fig6.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# 7) Feature importance: mean absolute coefficient across classes (for multiclass LR)
coefs = best_model.coef_
# if binary, coefs is shape (1, n_features), else (n_classes, n_features)
mean_abs_coef = np.mean(np.abs(coefs), axis=0)
feat_imp = pd.Series(mean_abs_coef, index=selected_features).sort_values(ascending=False)

fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax7)
ax7.set_title("Feature importance (mean abs coef across classes)")
ax7.set_xlabel("Mean |coefficient|")
plt.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))

# Compose a combined figure with some of these plots for convenience
fig_combined, axs = plt.subplots(3, 2, figsize=(14, 18))
axs = axs.flatten()

# 0: class dist before
sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=axs[0])
axs[0].set_title("Train distribution (before SMOTE)")

# 1: class dist after
sns.barplot(x=list(res_counts.keys()), y=list(res_counts.values()), ax=axs[1])
axs[1].set_title("Train distribution (after SMOTE)")

# 2: correlation heatmap small version
sns.heatmap(corr, annot=False, ax=axs[2])
axs[2].set_title("Feature correlation (overview)")

# 3: learning curve
axs[3].plot(train_sizes, train_scores_mean, marker="o", label="Train")
axs[3].plot(train_sizes, test_scores_mean, marker="o", label="CV")
axs[3].set_title("Learning curve")
axs[3].legend()
axs[3].grid(True)

# 4: confusion matrix
sns.heatmap(cm, annot=True, fmt="d", ax=axs[4])
axs[4].set_title("Confusion matrix")

# 5: top features bar
sns.barplot(x=feat_imp.values[:8], y=feat_imp.index[:8], ax=axs[5])
axs[5].set_title("Top features (mean abs coef)")

plt.tight_layout()
fig_combined.savefig(OUTPUT_PNG)

# Show plots (optional)
# plt.show()

print("Plots saved to:", OUTPUT_DIR)
print("Combined PNG:", OUTPUT_PNG)

# Save selected features and model info for reproducibility
meta = {
    "selected_features": selected_features,
    "best_params": grid.best_params_,
    "best_cv_score": float(grid.best_score_),
    "test_accuracy": float(acc)
}
pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "model_meta.json"))

# Save model using joblib (optional)
try:
    import joblib
    MODEL_PATH = os.path.join(OUTPUT_DIR, "logreg_model.joblib")
    joblib.dump({"model": best_model, "scaler": scaler, "features": selected_features}, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
except Exception as e:
    print("joblib save skipped:", e)

print("Done.")
