import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from itertools import product
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

from config import PARAM_GRID, PERM_REPEATS, SHAP_SAMPLE, OUTPUT_DIR
from utils import step_header, progress_bar


def prepare_data(df):
    feature_cols = [c for c in df.columns if c != "pattern"]
    X = df[feature_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df["pattern"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols, le


def tune(X_train, X_test, y_train, y_test, le):
    step_header(5, "Hyperparameter tuning")

    TUNE_SAMPLE = min(20_000, len(X_train))
    idx = np.random.choice(len(X_train), TUNE_SAMPLE, replace=False)
    X_tune = X_train.iloc[idx]
    y_tune = y_train[idx]

    print(f"  Grid search on {TUNE_SAMPLE:,} rows")

    results = []
    combos = list(product(PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"]))
    t0 = time.time()

    for i, (depth, lr) in enumerate(combos):
        m = XGBClassifier(
            n_estimators=100,
            max_depth=depth,
            learning_rate=lr,
            objective="multi:softmax",
            num_class=len(le.classes_),
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        m.fit(X_tune, y_tune)
        acc = (m.predict(X_test) == y_test).mean()
        results.append({"max_depth": depth, "learning_rate": lr, "accuracy": acc})
        progress_bar(i + 1, len(combos), task=f"depth={depth} lr={lr}  ", start_time=t0)

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    best = results_df.iloc[0]

    print(results_df.to_string(index=False))
    print(f"\n  Best params: max_depth={int(best.max_depth)}, learning_rate={best.learning_rate} (accuracy={best.accuracy:.4f})")
    return best


def train(X_train, X_test, y_train, y_test, best, le):
    step_header(6, "Train final XGBoost")
    t0 = time.time()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=int(best.max_depth),
        learning_rate=best.learning_rate,
        objective="multi:softmax",
        num_class=len(le.classes_),
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Training done in {time.time()-t0:.1f}s")
    return model


def evaluate(model, X_test, y_test, le):
    step_header(7, "Evaluate model")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.show()
    print("  Saved output/confusion_matrix.png")


def feature_importance(model, feature_cols, pattern_counts):
    step_header(8, "Feature importance")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(7, len(feature_cols) * 0.45)))
    fig.suptitle("Feature importance & Pattern frequency", fontsize=13)

    importances.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("XGBoost feature importance")
    axes[0].set_xlabel("Importance score")

    pattern_counts.sort_values().plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_title("Pattern frequency in dataset")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.show()
    print("  Saved output/feature_importance.png")

    return importances


def run_permutation(model, X_test, y_test, feature_cols, importances):
    step_header(9, "Permutation importance")

    PERM_SAMPLE = min(5_000, len(X_test))
    idx = np.random.choice(len(X_test), PERM_SAMPLE, replace=False)
    X_perm = X_test.iloc[idx]
    y_perm = y_test[idx]

    print(f"  Computing on {PERM_SAMPLE:,} rows with {PERM_REPEATS} repeats per feature...")
    t0 = time.time()

    perm = permutation_importance(
        model, X_perm, y_perm,
        n_repeats=PERM_REPEATS,
        random_state=42,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=True)

    print(f"  Done in {time.time()-t0:.1f}s\n")

    for _, row in perm_df.sort_values("importance", ascending=False).iterrows():
        bar = "█" * int(max(row["importance"], 0) * 200)
        print(f"  {row['feature']:<25} {bar:<40} {row['importance']:.4f} ± {row['std']:.4f}")

    # Permutation importance plot
    fig, ax = plt.subplots(figsize=(10, max(7, len(perm_df) * 0.45)))
    ax.barh(perm_df["feature"], perm_df["importance"], xerr=perm_df["std"],
            color="steelblue", ecolor="black", capsize=3)
    ax.set_title("Permutation importance\n(mean accuracy decrease when feature is randomly shuffled)")
    ax.set_xlabel("Mean accuracy decrease")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/permutation_importance.png", dpi=150)
    plt.show()
    print("  Saved output/permutation_importance.png")

    # Comparison plot
    fi_norm = importances / importances.max()
    pi = perm_df.set_index("feature")["importance"]
    pi_norm = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)

    compare_df = pd.DataFrame({
        "XGBoost importance (normalised)":     fi_norm,
        "Permutation importance (normalised)": pi_norm,
    }).sort_values("XGBoost importance (normalised)")

    fig, ax = plt.subplots(figsize=(10, max(7, len(compare_df) * 0.45)))
    compare_df.plot(kind="barh", ax=ax, color=["steelblue", "coral"])
    ax.set_title("Feature importance vs Permutation importance (normalised)")
    ax.set_xlabel("Normalised score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/importance_comparison.png", dpi=150)
    plt.show()
    print("  Saved output/importance_comparison.png")


def run_shap(model, X_test):
    step_header(10, "SHAP analysis")

    idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_shap = X_test.iloc[idx]

    print(f"  Computing SHAP values on {len(X_shap):,} samples...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    print(f"  Done in {time.time()-t0:.1f}s")

    sv_mean = np.array(shap_vals).mean(axis=2) if np.array(shap_vals).ndim == 3 else shap_vals
    shap.summary_plot(sv_mean, X_shap, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP — overall feature importance across all patterns")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_overall.png", dpi=120)
    plt.show()
    print("  Saved output/shap_overall.png")