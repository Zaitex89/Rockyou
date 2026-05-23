# RockYou Password Analyzer

XGBoost-based password pattern classifier trained on the RockYou dataset.
Performs EDA, feature extraction, hyperparameter tuning, and SHAP analysis.
Can be run as a command-line pipeline or as an interactive Streamlit web app.

---

## Project Structure

```
group_project6/
├── app.py           # Streamlit web app (interactive UI)
├── main.py          # CLI entry point — runs all steps in order
├── config.py        # All settings (sample size, paths, grid params)
├── utils.py         # Progress bar and step header helpers
├── data.py          # Steps 1-2: load passwords and EDA
├── features.py      # Steps 3-4: feature extraction and pattern classification
├── model.py         # Steps 5-10: tuning, training, evaluation, importance, SHAP
├── requirements.txt
├── rockyou.txt      # dataset (download separately, see below)
└── output/          # Created automatically — all PNGs saved here (CLI only)
```

---

## 1. Download the Dataset

Download `rockyou.txt` and place it in the project folder:

```
https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt
```

The file is ~133 MB and contains ~14 million real-world passwords.

---

## 2. Prerequisites

Make sure you have **Python 3.9 or higher** installed.
Check with:

```bash
python --version
```

---

## 3. Create a Virtual Environment

```bash
# Create the venv
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS / Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

This installs all dependencies including `pandas`, `numpy`, `matplotlib`, `seaborn`,
`xgboost`, `scikit-learn`, `shap`, and `streamlit`.

---

## 5. Run the Streamlit App

Make sure `rockyou.txt` is in the project folder, then run:

```bash
streamlit run app.py
```

A browser window will open automatically at `http://localhost:8501`.

The app has four tabs:

| Tab | Description |
|-----|-------------|
| Password Analysis | Enter any password to see its features, pattern class, and strength score. If a model has been trained, also shows ML classification with per-class probabilities. Includes a bulk analysis mode. |
| EDA | Load the dataset and generate interactive charts: length distribution, character types, entropy, pattern frequency, and a correlation heatmap. |
| Train Model | Run the full training pipeline in the browser: feature extraction, 3x3 grid search, final XGBoost model with 200 trees, confusion matrix, and classification report. |
| SHAP & Feature Importance | XGBoost feature importance, permutation importance with error bars, normalized comparison of both methods, and SHAP summary and beeswarm plots per pattern. |

Settings (sample size, SHAP samples, permutation repeats) are in the left sidebar.

---

## 6. Run the Command-Line Pipeline (alternative)

If you prefer to run without a browser:

```bash
python main.py
```

This runs all 10 steps in sequence and saves charts as PNG files to the `output/` folder.

| Step | Description |
|------|-------------|
| 1 | Load 100,000 random passwords from rockyou.txt |
| 2 | Exploratory Data Analysis — length, entropy, character types |
| 3 | Extract 22 features per password |
| 4 | Classify each password into a pattern (e.g. `word_number`, `only_letters`) |
| 5 | Grid search hyperparameter tuning (3x3 combos) |
| 6 | Train final XGBoost model with 200 trees |
| 7 | Evaluate — classification report and confusion matrix |
| 8 | XGBoost feature importance |
| 9 | Permutation importance |
| 10 | SHAP analysis |

Output files saved to `output/`:

| File | Description |
|------|-------------|
| `eda_plots.png` | Length distribution, char types, entropy, unique chars |
| `eda_correlation.png` | Correlation matrix of password features |
| `confusion_matrix.png` | Model prediction accuracy per pattern |
| `feature_importance.png` | XGBoost feature importance + pattern frequency |
| `permutation_importance.png` | Permutation importance with error bars |
| `importance_comparison.png` | XGBoost vs permutation importance (normalised) |
| `shap_overall.png` | SHAP summary across all patterns |

---

## 7. Configuration

Settings for the CLI pipeline are in `config.py`. The Streamlit app exposes the same
settings via the sidebar at runtime.

| Setting | Default | Description |
|---------|---------|-------------|
| `SAMPLE_SIZE` | `100_000` | Number of passwords to sample. Set to `None` to use all ~14M |
| `ROCKYOU_PATH` | `"rockyou.txt"` | Path to the dataset file |
| `SHAP_SAMPLE` | `1_000` | Number of samples for SHAP (higher = slower) |
| `PERM_REPEATS` | `5` | Repeats per feature for permutation importance |
| `OUTPUT_DIR` | `"output"` | Folder where CLI PNGs are saved |

---

## 8. Deactivate the Virtual Environment

When you're done:

```bash
deactivate
```
