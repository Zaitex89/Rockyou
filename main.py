import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import shap
import time
import sys


# CONFIG — change these before running


SAMPLE_SIZE = 100_000     # number of passwords to use
                          # set to None to use ALL passwords (~14M, slow)
                          # recommended: 100_000 for quick runs
                          #              500_000 for better accuracy
                          #              None    for full dataset

ROCKYOU_PATH = "rockyou.txt"

SHAP_SAMPLE  = 1_000      # how many rows to use for SHAP (keep low, it's slow)
PERM_REPEATS = 5          # how many times to shuffle each feature for permutation importance


# UTILITIES

def progress_bar(current, total, task="", bar_len=35, start_time=None):
    pct = current / total
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    elapsed = time.time() - start_time if start_time else 0
    eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
    eta_str = f"  ETA {eta:.0f}s" if 0 < pct < 1 else "        "
    sys.stdout.write(f"\r  [{bar}] {pct*100:5.1f}%  {task}{eta_str}")
    sys.stdout.flush()
    if current >= total:
        print()

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

STEPS = [
    "Load passwords",
    "Exploratory Data Analysis (EDA)",
    "Extract features",
    "Classify patterns",
    "Hyperparameter tuning",
    "Train final XGBoost",
    "Evaluate model",
    "Feature importance",
    "Permutation importance",
    "SHAP analysis",
]

def step_header(n, name):
    print(f"\n[{n}/{len(STEPS)}] {name}")
    print(f"  {'─'*50}")

# 1. LOAD PASSWORDS
step_header(1, STEPS[0])
t0 = time.time()

with open(ROCKYOU_PATH, "r", encoding="latin-1") as f:
    raw = f.readlines()

all_passwords = [line.strip() for line in raw if line.strip()]

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(all_passwords):
    # random sample so we get a representative mix
    np.random.seed(42)
    passwords = list(np.random.choice(all_passwords, SAMPLE_SIZE, replace=False))
    print(f"  Using random sample of {SAMPLE_SIZE:,} / {len(all_passwords):,} passwords")
else:
    passwords = all_passwords
    print(f"  Using ALL {len(passwords):,} passwords")

print(f"  Loaded in {time.time()-t0:.1f}s")


# 2. EDA — Exploratory Data Analysis
step_header(2, STEPS[1])

print("  Computing basic statistics...")

lengths      = [len(p) for p in passwords]
only_digits  = [p for p in passwords if p.isdigit()]
only_alpha   = [p for p in passwords if p.isalpha()]
has_special  = [p for p in passwords if any(not c.isalnum() for c in p)]
has_upper    = [p for p in passwords if any(c.isupper() for c in p)]

print(f"\n  Basic statistics")
print(f"  Total passwords        : {len(passwords):,}")
print(f"  Unique passwords       : {len(set(passwords)):,}  ({len(set(passwords))/len(passwords)*100:.1f}%)")
print(f"  Avg length             : {np.mean(lengths):.1f} chars")
print(f"  Most common length     : {max(set(lengths), key=lengths.count)}")
print(f"  Min / Max length       : {min(lengths)} / {max(lengths)}")
print(f"  Only digits            : {len(only_digits)/len(passwords)*100:.1f}%")
print(f"  Only letters           : {len(only_alpha)/len(passwords)*100:.1f}%")
print(f"  Contains special chars : {len(has_special)/len(passwords)*100:.1f}%")
print(f"  Contains uppercase     : {len(has_upper)/len(passwords)*100:.1f}%")

print(f"\n  Top 10 most common passwords")
top10 = pd.Series(passwords).value_counts().head(10)
for i, (pwd, cnt) in enumerate(top10.items(), 1):
    print(f"  {i:2}. {pwd:<25} {cnt:,} times")

# EDA plots
print("\n  Generating EDA plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"EDA — RockYou password dataset (n={len(passwords):,})", fontsize=14)

# 1. Length distribution
axes[0, 0].hist(
    [l for l in lengths if l <= 25],
    bins=25, color="steelblue", edgecolor="black"
)
axes[0, 0].axvline(x=8, color="red", linestyle="--", label="Min recommended (8)")
axes[0, 0].set_title("Password length distribution")
axes[0, 0].set_xlabel("Length")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# 2. Character type breakdown
char_labels = ["Only digits", "Only letters", "Has uppercase", "Has special"]
char_vals   = [
    len(only_digits)  / len(passwords) * 100,
    len(only_alpha)   / len(passwords) * 100,
    len(has_upper)    / len(passwords) * 100,
    len(has_special)  / len(passwords) * 100,
]
axes[0, 1].barh(char_labels, char_vals, color="coral")
axes[0, 1].set_title("Character type breakdown (%)")
axes[0, 1].set_xlabel("Percentage of passwords")

# 3. Top 15 passwords
top15 = pd.Series(passwords).value_counts().head(15)
axes[0, 2].barh(top15.index[::-1], top15.values[::-1], color="orange")
axes[0, 2].set_title("Top 15 most common passwords")
axes[0, 2].set_xlabel("Count")

# 4. Entropy distribution
entropies = []
for p in passwords[:50_000]:   # cap for speed
    if len(p) > 0:
        e = sum(-(p.count(c)/len(p)) * log2(p.count(c)/len(p)) for c in set(p))
        entropies.append(e)
axes[1, 0].hist(entropies, bins=40, color="green", edgecolor="black")
axes[1, 0].set_title("Password entropy distribution")
axes[1, 0].set_xlabel("Entropy (bits)")
axes[1, 0].set_ylabel("Count")

# 5. Length vs unique chars scatter
sample_idx = np.random.choice(len(passwords), min(5000, len(passwords)), replace=False)
s_lengths  = [len(passwords[i]) for i in sample_idx]
s_unique   = [len(set(passwords[i])) for i in sample_idx]
axes[1, 1].scatter(s_lengths, s_unique, alpha=0.2, s=5, color="purple")
axes[1, 1].set_title("Length vs unique characters")
axes[1, 1].set_xlabel("Password length")
axes[1, 1].set_ylabel("Unique characters")

# 6. Digit ratio distribution
digit_ratios = [sum(c.isdigit() for c in p) / len(p) for p in passwords[:50_000] if p]
axes[1, 2].hist(digit_ratios, bins=30, color="teal", edgecolor="black")
axes[1, 2].set_title("Digit ratio distribution")
axes[1, 2].set_xlabel("Fraction of digits in password")
axes[1, 2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()
print("  → Saved eda_plots.png")

# Correlation between numeric EDA features
print("\n  Computing feature correlations...")
eda_df = pd.DataFrame({
    "length":       [len(p) for p in passwords],
    "num_digits":   [sum(c.isdigit() for c in p) for p in passwords],
    "num_upper":    [sum(c.isupper() for c in p) for p in passwords],
    "num_special":  [sum(not c.isalnum() for c in p) for p in passwords],
    "unique_chars": [len(set(p)) for p in passwords],
})

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    eda_df.corr(), annot=True, fmt=".2f",
    cmap="coolwarm", center=0, ax=ax,
    linewidths=0.5
)
ax.set_title("Correlation matrix — password features")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()
print("  → Saved eda_correlation.png")

# 3. FEATURE EXTRACTION
step_header(3, STEPS[2])

def extract_features(pwd):
    if not pwd:
        return None
    return {
        "length":               len(pwd),
        "num_digits":           sum(c.isdigit() for c in pwd),
        "num_upper":            sum(c.isupper() for c in pwd),
        "num_lower":            sum(c.islower() for c in pwd),
        "num_special":          sum(not c.isalnum() for c in pwd),
        "digit_ratio":          sum(c.isdigit() for c in pwd) / len(pwd),
        "upper_ratio":          sum(c.isupper() for c in pwd) / len(pwd),
        "special_ratio":        sum(not c.isalnum() for c in pwd) / len(pwd),
        "entropy":              sum(
                                    -(pwd.count(c) / len(pwd)) * log2(pwd.count(c) / len(pwd))
                                    for c in set(pwd)
                                ),
        "unique_chars":         len(set(pwd)),
        "starts_with_upper":    int(pwd[0].isupper()),
        "ends_with_digit":      int(pwd[-1].isdigit()),
        "ends_with_special":    int(not pwd[-1].isalnum()),
        "only_digits":          int(pwd.isdigit()),
        "only_lower":           int(pwd.islower()),
        "only_upper":           int(pwd.isupper()),
        "only_alpha":           int(pwd.isalpha()),
        "has_year":             int(bool(re.search(r"(19|20)\d{2}", pwd))),
        "has_keyboard_walk":    int(bool(re.search(
                                    r"qwer|asdf|zxcv|1234|2345|3456|4567", pwd.lower()
                                ))),
        "has_repeated_chars":   int(bool(re.search(r"(.)\1{2,}", pwd))),
        "is_leet":              int(bool(re.search(r"[4310!@$]", pwd))),
        "capital_lower_digit":  int(bool(re.match(r"^[A-Z][a-z]+\d+$", pwd))),
    }

t0 = time.time()
feature_rows = []
skipped = 0

for i, pwd in enumerate(passwords):
    feats = extract_features(pwd)
    if feats is None:
        skipped += 1
    else:
        feature_rows.append(feats)
    if (i + 1) % max(1, len(passwords) // 100) == 0 or i + 1 == len(passwords):
        progress_bar(i + 1, len(passwords), task="extracting  ", start_time=t0)

print(f"  → {len(feature_rows):,} processed, {skipped:,} skipped ({time.time()-t0:.1f}s)")

# 4. CLASSIFY PATTERNS (labels)
step_header(4, STEPS[3])

def classify_pattern(pwd):
    if not pwd:                                   return "empty"
    if re.match(r"^\d+$", pwd):                   return "only_numbers"
    if re.match(r"^[a-zA-Z]+$", pwd):             return "only_letters"
    if re.match(r"^[A-Z][a-z]+\d+$", pwd):        return "name_number"
    if re.match(r"^[a-z]+\d+$", pwd):             return "word_number"
    if re.match(r"^[a-z]+[!@#$%^&*]+$", pwd):    return "word_special"
    if re.match(r"^[a-z]+\d+[!@#$%^&*]+$", pwd): return "word_number_special"
    if re.search(r"(19|20)\d{2}", pwd):           return "contains_year"
    if re.search(r"qwer|asdf|1234|abcd",
                 pwd.lower()):                    return "keyboard_walk"
    if re.search(r"(.)\1{2,}", pwd):              return "repeated_chars"
    return "other"

t0 = time.time()
labels = []
valid_passwords = [p for p in passwords if extract_features(p) is not None]

for i, pwd in enumerate(valid_passwords):
    labels.append(classify_pattern(pwd))
    if (i + 1) % max(1, len(valid_passwords) // 100) == 0 or i + 1 == len(valid_passwords):
        progress_bar(i + 1, len(valid_passwords), task="classifying ", start_time=t0)

print(f"  → Labelled {len(labels):,} passwords ({time.time()-t0:.1f}s)")

df = pd.DataFrame(feature_rows)
df["pattern"] = labels

# Pattern distribution
print(f"\n  Pattern distribution")
pattern_counts = df["pattern"].value_counts()
total = len(df)
for pat, cnt in pattern_counts.items():
    bar = "█" * int(cnt / total * 40)
    print(f"  {pat:<22} {bar:<42} {cnt/total*100:5.1f}%  ({cnt:,})")

# 5. HYPERPARAMETER TUNING (grid search on small sample)
step_header(5, STEPS[4])

feature_cols = [c for c in df.columns if c != "pattern"]
X = df[feature_cols].fillna(0)
le = LabelEncoder()
y = le.fit_transform(df["pattern"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use a small subsample for grid search so it doesn't take forever
TUNE_SAMPLE = min(20_000, len(X_train))
idx = np.random.choice(len(X_train), TUNE_SAMPLE, replace=False)
X_tune = X_train.iloc[idx]
y_tune = y_train[idx]

print(f"  Grid search on {TUNE_SAMPLE:,} rows (subset of training data)")
print(f"  Testing combinations of max_depth and learning_rate...\n")

param_grid = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}

# Manual grid search with progress so we can show % 
from itertools import product

results = []
combos = list(product(param_grid["max_depth"], param_grid["learning_rate"]))
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
    results.append({
        "max_depth": depth,
        "learning_rate": lr,
        "accuracy": acc,
    })
    progress_bar(i + 1, len(combos), task=f"depth={depth} lr={lr}  ", start_time=t0)

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
best = results_df.iloc[0]

print(f"\n  uning results")
print(results_df.to_string(index=False))
print(f"\n  Best params → max_depth={int(best.max_depth)}, learning_rate={best.learning_rate} (accuracy={best.accuracy:.4f})")


# 6. TRAIN FINAL MODEL with best params
step_header(6, STEPS[5])
print(f"  Training with best hyperparameters on full training set...")
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

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print(f"  → Training done in {time.time()-t0:.1f}s")

# 7. EVALUATE MODEL
step_header(7, STEPS[6])

y_pred = model.predict(X_test)
print("\n  Classification report")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
ax.set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("  → Saved confusion_matrix.png")

# 8. FEATURE IMPORTANCE
step_header(8, STEPS[7])

importances = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature importance & Pattern frequency", fontsize=13)

importances.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("XGBoost feature importance\n(how much each feature was used in trees)")
axes[0].set_xlabel("Importance score")

pattern_counts.sort_values().plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("Pattern frequency in dataset\n(how common each pattern is)")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("  → Saved feature_importance.png")

# 9. PERMUTATION IMPORTANCE
step_header(9, STEPS[8])

# Use a subsample of test set so it doesn't take too long
PERM_SAMPLE = min(5_000, len(X_test))
idx = np.random.choice(len(X_test), PERM_SAMPLE, replace=False)
X_perm = X_test.iloc[idx]
y_perm = y_test[idx]

print(f"  Computing on {PERM_SAMPLE:,} rows with {PERM_REPEATS} repeats per feature...")
print(f"  (shuffles each feature {PERM_REPEATS}x and measures accuracy drop)")
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

print(f"  → Done in {time.time()-t0:.1f}s")

print(f"\n  Permutation importance (top features)")
for _, row in perm_df.sort_values("importance", ascending=False).head(10).iterrows():
    bar = "█" * int(max(row["importance"], 0) * 200)
    print(f"  {row['feature']:<25} {bar:<40} {row['importance']:.4f} ± {row['std']:.4f}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    perm_df["feature"],
    perm_df["importance"],
    xerr=perm_df["std"],
    color="steelblue",
    ecolor="black",
    capsize=3,
)
ax.set_title(
    "Permutation importance\n"
    "(mean accuracy decrease when feature is randomly shuffled)"
)
ax.set_xlabel("Mean accuracy decrease")
ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150)
plt.show()
print("  → Saved permutation_importance.png")

# Side-by-side comparison: feature importance vs permutation importance
fi_norm = importances / importances.max()
pi = perm_df.set_index("feature")["importance"]
pi_norm = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)

compare_df = pd.DataFrame({
    "XGBoost importance (normalised)":    fi_norm,
    "Permutation importance (normalised)": pi_norm,
}).sort_values("XGBoost importance (normalised)")

fig, ax = plt.subplots(figsize=(10, 7))
compare_df.plot(kind="barh", ax=ax, color=["steelblue", "coral"])
ax.set_title("Feature importance vs Permutation importance (normalised)")
ax.set_xlabel("Normalised score")
plt.tight_layout()
plt.savefig("importance_comparison.png", dpi=150)
plt.show()
print("  → Saved importance_comparison.png")

# 10. SHAP
step_header(10, STEPS[9])

idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
X_shap = X_test.iloc[idx]

print(f"  Computing SHAP values on {len(X_shap):,} samples...")
t0 = time.time()
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_shap)
print(f"  → Done in {time.time()-t0:.1f}s")

# One bar plot per pattern class
for i, class_name in enumerate(le.classes_):
    progress_bar(i + 1, len(le.classes_), task=f"plotting {class_name:<18}")
    sv = shap_vals[:, :, i] if np.array(shap_vals).ndim == 3 else shap_vals[i]
    shap.summary_plot(sv, X_shap, plot_type="bar", show=False, max_display=10)
    plt.title(f"SHAP — top features driving pattern: '{class_name}'")
    plt.tight_layout()
    plt.savefig(f"shap_{class_name}.png", dpi=120)
    plt.close()

# FINAL SUMMARY
section("All done! Output files")
output_files = (
    ["eda_plots.png", "eda_correlation.png", "confusion_matrix.png",
     "feature_importance.png", "permutation_importance.png",
     "importance_comparison.png"]
    + [f"shap_{c}.png" for c in le.classes_]
)
for f in output_files:
    print(f"  {f}")

print(f"\n  Sample size used : {len(passwords):,}")
print(f"  Patterns found   : {len(le.classes_)}")
print(f"  Best params      : max_depth={int(best.max_depth)}, lr={best.learning_rate}")