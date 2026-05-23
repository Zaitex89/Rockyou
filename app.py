"""
RockYou Password Analyzer - Streamlit App
Run with: streamlit run app.py
"""

import os
import sys
import time
import random
import re
from math import log2
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import streamlit as st

# page config
st.set_page_config(
    page_title="RockYou Password Analyzer",
    layout="wide",
)

# config - mirrors config.py but editable via sidebar
ROCKYOU_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "rockyou.txt")
PARAM_GRID = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}

# feature extraction helpers
def extract_features(pwd):
    if not pwd:
        return None
    return {
        "length":              len(pwd),
        "num_digits":          sum(c.isdigit() for c in pwd),
        "num_upper":           sum(c.isupper() for c in pwd),
        "num_lower":           sum(c.islower() for c in pwd),
        "num_special":         sum(not c.isalnum() for c in pwd),
        "digit_ratio":         sum(c.isdigit() for c in pwd) / len(pwd),
        "upper_ratio":         sum(c.isupper() for c in pwd) / len(pwd),
        "special_ratio":       sum(not c.isalnum() for c in pwd) / len(pwd),
        "entropy":             sum(
                                   -(pwd.count(c) / len(pwd)) * log2(pwd.count(c) / len(pwd))
                                   for c in set(pwd)
                               ),
        "unique_chars":        len(set(pwd)),
        "starts_with_upper":   int(pwd[0].isupper()),
        "ends_with_digit":     int(pwd[-1].isdigit()),
        "ends_with_special":   int(not pwd[-1].isalnum()),
        "only_digits":         int(pwd.isdigit()),
        "only_lower":          int(pwd.islower()),
        "only_upper":          int(pwd.isupper()),
        "only_alpha":          int(pwd.isalpha()),
        "has_year":            int(bool(re.search(r"(19|20)\d{2}", pwd))),
        "has_keyboard_walk":   int(bool(re.search(r"qwer|asdf|zxcv|1234|2345|3456|4567", pwd.lower()))),
        "has_repeated_chars":  int(bool(re.search(r"(.)\1{2,}", pwd))),
        "is_leet":             int(bool(re.search(r"[4310!@$]", pwd))),
        "capital_lower_digit": int(bool(re.match(r"^[A-Z][a-z]+\d+$", pwd))),
    }


def classify_pattern(pwd):
    if not pwd:                                         return "empty"
    if re.match(r"^\d+$", pwd):                        return "only_numbers"
    if re.match(r"^[a-zA-Z]+$", pwd):                  return "only_letters"
    if re.match(r"^[A-Z][a-z]+\d+$", pwd):             return "name_number"
    if re.match(r"^[a-z]+\d+$", pwd):                  return "word_number"
    if re.match(r"^[a-z]+[!@#$%^&*]+$", pwd):          return "word_special"
    if re.match(r"^[a-z]+\d+[!@#$%^&*]+$", pwd):       return "word_number_special"
    if re.search(r"(19|20)\d{2}", pwd):                 return "contains_year"
    if re.search(r"qwer|asdf|1234|abcd", pwd.lower()):  return "keyboard_walk"
    if re.search(r"(.)\1{2,}", pwd):                    return "repeated_chars"
    return "other"


PATTERN_DESCRIPTIONS = {
    "only_numbers":        "Digits only - e.g. 123456",
    "only_letters":        "Letters only - e.g. password",
    "name_number":         "Name followed by digits - e.g. Alice2023",
    "word_number":         "Word followed by digits - e.g. hello99",
    "word_special":        "Word followed by special characters - e.g. hello!!",
    "word_number_special": "Word, digits, then special characters - e.g. hi5!",
    "contains_year":       "Contains a year pattern - e.g. pass1995",
    "keyboard_walk":       "Keyboard walk sequence - e.g. qwerty",
    "repeated_chars":      "Repeated character sequence - e.g. aaabbb",
    "other":               "Mixed or unclassified pattern",
}

STRENGTH_THRESHOLDS = [
    (3.0, "Very weak", "red"),
    (2.0, "Weak", "orange"),
    (1.5, "Medium", "gold"),
    (1.0, "Good", "green"),
    (0.0, "Strong", "blue"),
]


def password_strength(feats):
    """Return (label, color) based on entropy and feature mix."""
    ent = feats["entropy"]
    length = feats["length"]
    has_variety = (feats["num_digits"] > 0 and feats["num_upper"] > 0
                   and feats["num_lower"] > 0 and feats["num_special"] > 0)
    score = ent * (1 + 0.3 * (length >= 12)) * (1 + 0.2 * has_variety)
    if score >= 3.0:   return "Strong", "green"
    if score >= 2.0:   return "Good", "#90EE90"
    if score >= 1.2:   return "Medium", "gold"
    if score >= 0.5:   return "Weak", "orange"
    return "Very weak", "red"


# cached data loading
@st.cache_data(show_spinner=False)
def load_passwords(path, sample_size):
    with open(path, "r", encoding="latin-1") as f:
        raw = f.readlines()
    all_passwords = [line.strip() for line in raw if line.strip()]
    if sample_size and sample_size < len(all_passwords):
        random.seed(42)
        return random.sample(all_passwords, sample_size), len(all_passwords)
    return all_passwords, len(all_passwords)


@st.cache_data(show_spinner=False)
def build_dataframe(passwords_tuple):
    passwords = list(passwords_tuple)
    feature_rows, labels, valid_pwds = [], [], []
    for pwd in passwords:
        feats = extract_features(pwd)
        if feats is not None:
            feature_rows.append(feats)
            labels.append(classify_pattern(pwd))
            valid_pwds.append(pwd)
    df = pd.DataFrame(feature_rows)
    df["pattern"] = labels
    return df


@st.cache_resource(show_spinner=False)
def train_model(df_hash, sample_size, shap_sample, perm_repeats):
    """Run the full training pipeline and return model artifacts."""
    # df is passed via session_state to avoid hashing large objects
    df = st.session_state["_df_for_training"]

    feature_cols = [c for c in df.columns if c != "pattern"]
    X = df[feature_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df["pattern"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # grid search on a subsample for speed
    TUNE_SAMPLE = min(20_000, len(X_train))
    idx = np.random.choice(len(X_train), TUNE_SAMPLE, replace=False)
    X_tune, y_tune = X_train.iloc[idx], y_train[idx]

    combos = list(product(PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"]))
    results = []
    for depth, lr in combos:
        m = XGBClassifier(
            n_estimators=100, max_depth=depth, learning_rate=lr,
            objective="multi:softmax", num_class=len(le.classes_),
            tree_method="hist", n_jobs=-1, random_state=42,
            eval_metric="mlogloss", verbosity=0,
        )
        m.fit(X_tune, y_tune)
        acc = (m.predict(X_test) == y_test).mean()
        results.append({"max_depth": depth, "learning_rate": lr, "accuracy": acc})

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    best = results_df.iloc[0]

    # train final model with best hyperparameters
    model = XGBClassifier(
        n_estimators=200, max_depth=int(best.max_depth),
        learning_rate=best.learning_rate,
        objective="multi:softmax", num_class=len(le.classes_),
        tree_method="hist", n_jobs=-1, random_state=42,
        eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return model, le, feature_cols, X_train, X_test, y_train, y_test, results_df, best


# sidebar settings
with st.sidebar:
    st.title("Settings")
    rockyou_path = st.text_input("Path to rockyou.txt", value=ROCKYOU_PATH_DEFAULT)
    sample_size = st.number_input(
        "Sample size (0 = all ~14M)", min_value=0, max_value=14_000_000,
        value=100_000, step=10_000
    )
    sample_size = int(sample_size) if sample_size > 0 else None
    shap_sample = st.slider("SHAP sample size", 200, 5000, 1000, step=100)
    perm_repeats = st.slider("Permutation repeats", 1, 10, 5)

    st.divider()
    st.caption("RockYou Password Analyzer · XGBoost + SHAP")

# tabs
tab_pwd, tab_eda, tab_train, tab_shap = st.tabs([
    "Password Analysis",
    "EDA",
    "Train Model",
    "SHAP & Feature Importance",
])


# TAB 1 - Password Analysis
with tab_pwd:
    st.header("Password Analysis")
    st.write(
        "Enter any password to see its extracted features, "
        "pattern class, and a strength estimate."
    )

    col_input, col_compare = st.columns([2, 1])
    with col_input:
        pwd_input = st.text_input(
            "Password to analyze",
            type="password",
            placeholder="Type a password...",
        )
        show_pwd = st.checkbox("Show password in plain text")
        if show_pwd and pwd_input:
            st.code(pwd_input)

    if pwd_input:
        feats = extract_features(pwd_input)
        pattern = classify_pattern(pwd_input)
        strength_label, strength_color = password_strength(feats)

        st.divider()

        # top-level metrics
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Pattern", pattern)
        col_s2.metric("Entropy", f"{feats['entropy']:.2f} bits")
        col_s3.metric("Strength", strength_label)

        # pattern description
        desc = PATTERN_DESCRIPTIONS.get(pattern, "")
        st.info(f"**Pattern:** {desc}")

        st.subheader("All features")
        feat_df = pd.DataFrame([feats]).T.rename(columns={0: "Value"})
        feat_df.index.name = "Feature"
        feat_df["Value"] = feat_df["Value"].round(4)

        # highlight boolean features green/red
        bool_feats = [
            "starts_with_upper", "ends_with_digit", "ends_with_special",
            "only_digits", "only_lower", "only_upper", "only_alpha",
            "has_year", "has_keyboard_walk", "has_repeated_chars",
            "is_leet", "capital_lower_digit",
        ]

        def highlight_bool(row):
            styles = []
            for val in row:
                if row.name in bool_feats:
                    styles.append("background-color: #d4edda" if val == 1 else "background-color: #f8d7da")
                else:
                    styles.append("")
            return styles

        st.dataframe(
            feat_df.style.apply(highlight_bool, axis=1),
            use_container_width=True,
        )

        # show model prediction if a model has been trained
        if "trained_model" in st.session_state:
            st.subheader("Model classification")
            model = st.session_state["trained_model"]
            le    = st.session_state["trained_le"]
            feature_cols = st.session_state["trained_feature_cols"]
            feat_series = pd.DataFrame([feats])[feature_cols].fillna(0)
            pred_idx = model.predict(feat_series)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            proba = model.predict_proba(feat_series)[0]
            top_n = np.argsort(proba)[::-1][:5]

            st.success(f"Model classifies this password as: **{pred_label}**")
            proba_df = pd.DataFrame({
                "Pattern":     le.inverse_transform(top_n),
                "Probability": proba[top_n],
            })
            st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
        else:
            st.info("Train the model in the **Train Model** tab to also get an ML-based classification.")

    # bulk analysis section
    with st.expander("Analyze multiple passwords at once"):
        bulk_input = st.text_area(
            "Paste passwords, one per line",
            height=120,
            placeholder="password123\nqwerty\nSuperSecret!2024",
        )
        if bulk_input.strip():
            pwds = [p.strip() for p in bulk_input.strip().splitlines() if p.strip()]
            rows = []
            for p in pwds:
                f = extract_features(p)
                if f:
                    sl, _ = password_strength(f)
                    rows.append({
                        "Password": p,
                        "Pattern":  classify_pattern(p),
                        "Length":   f["length"],
                        "Entropy":  round(f["entropy"], 2),
                        "Strength": sl,
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


# TAB 2 - EDA
with tab_eda:
    st.header("Exploratory Data Analysis")

    if not os.path.exists(rockyou_path):
        st.error(f"File not found: `{rockyou_path}`. Check the path in the sidebar.")
    else:
        if st.button("Load data and generate EDA", key="eda_btn"):
            with st.spinner("Loading passwords..."):
                passwords, total = load_passwords(rockyou_path, sample_size)
                st.session_state["passwords"] = passwords
                st.session_state["total_passwords"] = total

        passwords = st.session_state.get("passwords")

        if passwords:
            total = st.session_state["total_passwords"]
            n = len(passwords)
            st.success(f"Loaded **{n:,}** passwords (out of **{total:,}** total in the dataset)")

            # summary statistics
            lengths     = [len(p) for p in passwords]
            only_digits = [p for p in passwords if p.isdigit()]
            only_alpha  = [p for p in passwords if p.isalpha()]
            has_special = [p for p in passwords if any(not c.isalnum() for c in p)]
            has_upper   = [p for p in passwords if any(c.isupper() for c in p)]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mean length",         f"{np.mean(lengths):.1f}")
            c2.metric("Most common length",  str(max(set(lengths), key=lengths.count)))
            c3.metric("Digits only",         f"{len(only_digits)/n*100:.1f}%")
            c4.metric("Letters only",        f"{len(only_alpha)/n*100:.1f}%")
            c5.metric("Has special chars",   f"{len(has_special)/n*100:.1f}%")

            # 2x2 EDA plot
            with st.spinner("Generating plots..."):
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"EDA - RockYou dataset (n={n:,})", fontsize=14)

                axes[0, 0].hist([l for l in lengths if l <= 25], bins=25,
                                color="steelblue", edgecolor="black")
                axes[0, 0].axvline(x=8, color="red", linestyle="--", label="Min recommended (8)")
                axes[0, 0].set_title("Password length distribution")
                axes[0, 0].set_xlabel("Length")
                axes[0, 0].set_ylabel("Count")
                axes[0, 0].legend()

                char_labels = ["Digits only", "Letters only", "Has uppercase", "Has special"]
                char_vals = [
                    len(only_digits) / n * 100,
                    len(only_alpha)  / n * 100,
                    len(has_upper)   / n * 100,
                    len(has_special) / n * 100,
                ]
                axes[0, 1].barh(char_labels, char_vals, color="coral")
                axes[0, 1].set_title("Character type breakdown (%)")
                axes[0, 1].set_xlabel("Percentage of passwords")

                entropies = []
                for p in passwords[:50_000]:
                    if p:
                        e = sum(-(p.count(c) / len(p)) * log2(p.count(c) / len(p)) for c in set(p))
                        entropies.append(e)
                axes[1, 0].hist(entropies, bins=40, color="green", edgecolor="black")
                axes[1, 0].set_title("Entropy distribution")
                axes[1, 0].set_xlabel("Entropy (bits)")

                sample_idx = np.random.choice(n, min(5000, n), replace=False)
                s_len  = [len(passwords[i]) for i in sample_idx]
                s_uniq = [len(set(passwords[i])) for i in sample_idx]
                axes[1, 1].scatter(s_len, s_uniq, alpha=0.2, s=5, color="purple")
                axes[1, 1].set_title("Length vs unique characters")
                axes[1, 1].set_xlabel("Length")
                axes[1, 1].set_ylabel("Unique characters")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # pattern frequency bar chart
            with st.spinner("Counting patterns..."):
                pattern_counts = pd.Series([classify_pattern(p) for p in passwords]).value_counts()
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                pattern_counts.sort_values().plot(kind="barh", ax=ax2, color="steelblue")
                ax2.set_title("Pattern distribution in dataset")
                ax2.set_xlabel("Number of passwords")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            # correlation heatmap
            with st.spinner("Computing correlation matrix..."):
                eda_df = pd.DataFrame({
                    "length":       [len(p) for p in passwords],
                    "num_digits":   [sum(c.isdigit() for c in p) for p in passwords],
                    "num_upper":    [sum(c.isupper() for c in p) for p in passwords],
                    "num_special":  [sum(not c.isalnum() for c in p) for p in passwords],
                    "unique_chars": [len(set(p)) for p in passwords],
                })
                fig3, ax3 = plt.subplots(figsize=(7, 5))
                sns.heatmap(eda_df.corr(), annot=True, fmt=".2f",
                            cmap="coolwarm", center=0, ax=ax3, linewidths=0.5)
                ax3.set_title("Correlation matrix")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
        else:
            st.info("Click **Load data and generate EDA** to get started.")


# TAB 3 - Train Model
with tab_train:
    st.header("Train XGBoost Model")
    st.write(
        "Run the full training pipeline: feature extraction, grid search, "
        "final model training, and evaluation."
    )

    if not os.path.exists(rockyou_path):
        st.error(f"File not found: `{rockyou_path}`. Check the path in the sidebar.")
    else:
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            run_training = st.button("Start training", key="train_btn", type="primary")
        with col_info:
            st.caption(
                f"Sample: {sample_size:,} passwords | "
                f"Grid: {len(PARAM_GRID['max_depth']) * len(PARAM_GRID['learning_rate'])} combinations"
                if sample_size else
                "Sample: all passwords (this may take a while)"
            )

        if run_training:
            progress = st.progress(0, text="Loading passwords...")

            # step 1: load passwords
            with st.spinner("Step 1/6 - Loading passwords..."):
                passwords, total = load_passwords(rockyou_path, sample_size)
                st.session_state["passwords"] = passwords
            progress.progress(10, text="Step 2/6 - Extracting features...")

            # step 2: build feature dataframe
            with st.spinner("Step 2/6 - Extracting features..."):
                df = build_dataframe(tuple(passwords))
                st.session_state["_df_for_training"] = df
            progress.progress(30, text="Step 3/6 - Running grid search...")

            # steps 3-6: grid search and training
            with st.spinner("Steps 3-6 - Grid search and training..."):
                cache_key = (sample_size or 0, shap_sample, perm_repeats)
                result = train_model(cache_key, sample_size, shap_sample, perm_repeats)
                model, le, feature_cols, X_train, X_test, y_train, y_test, results_df, best = result

            progress.progress(80, text="Step 7/6 - Evaluating model...")
            st.session_state["trained_model"]        = model
            st.session_state["trained_le"]           = le
            st.session_state["trained_feature_cols"] = feature_cols
            st.session_state["trained_X_test"]       = X_test
            st.session_state["trained_y_test"]       = y_test
            st.session_state["trained_df"]           = df
            st.session_state["training_done"]        = True
            progress.progress(100, text="Done!")
            st.success("Training complete!")

        if st.session_state.get("training_done"):
            model        = st.session_state["trained_model"]
            le           = st.session_state["trained_le"]
            feature_cols = st.session_state["trained_feature_cols"]
            X_test       = st.session_state["trained_X_test"]
            y_test       = st.session_state["trained_y_test"]
            df           = st.session_state["trained_df"]

            # retrieve cached results
            result = train_model(
                (sample_size or 0, shap_sample, perm_repeats),
                sample_size, shap_sample, perm_repeats
            )
            _, _, _, X_train, X_test, y_train, y_test, results_df, best = result

            st.subheader("Grid search results")
            st.dataframe(
                results_df.style.highlight_max(subset=["accuracy"], color="#074014")
                          .format({"accuracy": "{:.4f}"}),
                use_container_width=True,
            )
            st.info(
                f"Best parameters: **max_depth={int(best.max_depth)}**, "
                f"**learning_rate={best.learning_rate}** "
                f"(accuracy={best.accuracy:.4f})"
            )

            # confusion matrix
            st.subheader("Confusion matrix")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(ax=ax_cm, xticks_rotation=45, colorbar=True)
            ax_cm.set_title("Confusion matrix")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # classification report
            with st.expander("Classification report"):
                report = classification_report(
                    y_test, y_pred, target_names=le.classes_, output_dict=True
                )
                st.dataframe(
                    pd.DataFrame(report).T.round(3),
                    use_container_width=True,
                )
        else:
            st.info("Click **Start training** to run the pipeline.")


# TAB 4 - SHAP and Feature Importance
with tab_shap:
    st.header("SHAP & Feature Importance")

    if not st.session_state.get("training_done"):
        st.info("Train the model first in the **Train Model** tab.")
    else:
        model        = st.session_state["trained_model"]
        le           = st.session_state["trained_le"]
        feature_cols = st.session_state["trained_feature_cols"]
        X_test       = st.session_state["trained_X_test"]
        y_test       = st.session_state["trained_y_test"]
        df           = st.session_state["trained_df"]

        pattern_counts = df["pattern"].value_counts()
        importances    = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

        # XGBoost built-in feature importance
        st.subheader("XGBoost Feature Importance")
        fig_fi, axes_fi = plt.subplots(1, 2, figsize=(16, max(7, len(feature_cols) * 0.45)))
        importances.plot(kind="barh", ax=axes_fi[0], color="steelblue")
        axes_fi[0].set_title("XGBoost feature importance")
        axes_fi[0].set_xlabel("Importance score")

        pattern_counts.sort_values().plot(kind="barh", ax=axes_fi[1], color="coral")
        axes_fi[1].set_title("Pattern frequency in dataset")
        axes_fi[1].set_xlabel("Count")

        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        # permutation importance
        st.subheader("Permutation Importance")
        with st.spinner(f"Computing permutation importance ({perm_repeats} repeats)..."):
            PERM_SAMPLE = min(5_000, len(X_test))
            idx_perm = np.random.choice(len(X_test), PERM_SAMPLE, replace=False)
            X_perm   = X_test.iloc[idx_perm]
            y_perm   = y_test[idx_perm]

            perm = permutation_importance(
                model, X_perm, y_perm,
                n_repeats=perm_repeats,
                random_state=42,
                n_jobs=-1,
            )
            perm_df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": perm.importances_mean,
                "std":        perm.importances_std,
            }).sort_values("importance", ascending=True)

        fig_perm, ax_perm = plt.subplots(figsize=(10, max(7, len(perm_df) * 0.45)))
        ax_perm.barh(
            perm_df["feature"], perm_df["importance"],
            xerr=perm_df["std"], color="steelblue", ecolor="black", capsize=3
        )
        ax_perm.set_title("Permutation importance")
        ax_perm.set_xlabel("Mean accuracy decrease")
        ax_perm.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_perm)
        plt.close(fig_perm)

        # normalized comparison of both importance methods
        st.subheader("XGBoost vs Permutation importance (normalized)")
        fi_norm = importances / importances.max()
        pi      = perm_df.set_index("feature")["importance"]
        pi_norm = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)
        compare_df = pd.DataFrame({
            "XGBoost (norm.)":     fi_norm,
            "Permutation (norm.)": pi_norm,
        }).sort_values("XGBoost (norm.)")

        fig_cmp, ax_cmp = plt.subplots(figsize=(10, max(7, len(compare_df) * 0.45)))
        compare_df.plot(kind="barh", ax=ax_cmp, color=["steelblue", "coral"])
        ax_cmp.set_title("Feature importance vs permutation importance")
        ax_cmp.set_xlabel("Normalized score")
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        # SHAP summary
        st.subheader("SHAP Analysis")
        with st.spinner(f"Computing SHAP values on {shap_sample} samples..."):
            idx_shap = np.random.choice(len(X_test), min(shap_sample, len(X_test)), replace=False)
            X_shap   = X_test.iloc[idx_shap]

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_shap)

        sv = np.array(shap_vals)
        sv_mean = sv.mean(axis=2) if sv.ndim == 3 else sv

        fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv_mean, X_shap, plot_type="bar", show=False, max_display=15)
        plt.title("SHAP - overall feature importance across all patterns")
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        # per-class beeswarm plot
        with st.expander("SHAP beeswarm by pattern"):
            selected_class = st.selectbox("Select pattern", le.classes_)
            class_idx = list(le.classes_).index(selected_class)

            if sv.ndim == 3:
                # newer SHAP returns (samples, features, classes), older returns (classes, samples, features)
                if sv.shape[0] == len(le.classes_):
                    sv_class = sv[class_idx]        # old format: (samples, features)
                else:
                    sv_class = sv[:, :, class_idx]  # new format: (samples, features)
            else:
                sv_class = sv

            fig_bee, ax_bee = plt.subplots(figsize=(10, 7))
            shap.summary_plot(sv_class, X_shap, show=False, max_display=15)
            plt.title(f"SHAP beeswarm - pattern: {selected_class}")
            plt.tight_layout()
            st.pyplot(fig_bee)
            plt.close(fig_bee)
