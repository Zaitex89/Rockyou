# RockYou Password Analyzer

XGBoost-based password pattern classifier trained on the RockYou dataset.
Performs EDA, feature extraction, hyperparameter tuning, and SHAP analysis.

---

## Project Structure

```
rockyou/
├── main.py          # Entry point — runs all steps in order
├── config.py        # All settings (sample size, paths, grid params)
├── utils.py         # Progress bar and step header helpers
├── data.py          # Steps 1–2: load passwords and EDA
├── features.py      # Steps 3–4: feature extraction and pattern classification
├── model.py         # Steps 5–10: tuning, training, evaluation, importance, SHAP
├── requirements.txt
└── output/          # Created automatically — all PNGs saved here
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

This installs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`, and `shap`.

---

## 5. Run the Program

Make sure your folder looks like this before running:

```
rockyou/
├── main.py
├── config.py
├── utils.py
├── data.py
├── features.py
├── model.py
├── requirements.txt
└── rockyou.txt        ← must be here
```

Then run:

```bash
python main.py
```

---

## 6. What It Does

The program runs 10 steps automatically:

| Step | Description |
|------|-------------|
| 1 | Load 100,000 random passwords from rockyou.txt |
| 2 | Exploratory Data Analysis — length, entropy, character types |
| 3 | Extract 22 features per password |
| 4 | Classify each password into a pattern (e.g. `word_number`, `only_letters`) |
| 5 | Grid search hyperparameter tuning (3×3 combos) |
| 6 | Train final XGBoost model with 200 trees |
| 7 | Evaluate — classification report and confusion matrix |
| 8 | XGBoost feature importance |
| 9 | Permutation importance |
| 10 | SHAP analysis |

---

## 7. Output

All PNG charts are saved to the `output/` folder:

| File | Description |
|------|-------------|
| `output/eda_plots.png` | Length distribution, char types, entropy, unique chars |
| `output/eda_correlation.png` | Correlation matrix of password features |
| `output/confusion_matrix.png` | Model prediction accuracy per pattern |
| `output/feature_importance.png` | XGBoost feature importance + pattern frequency |
| `output/permutation_importance.png` | Permutation importance with error bars |
| `output/importance_comparison.png` | XGBoost vs permutation importance (normalised) |
| `output/shap_overall.png` | SHAP summary across all patterns |

---

## 8. Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SAMPLE_SIZE` | `100_000` | Number of passwords to sample. Set to `None` to use all ~14M |
| `ROCKYOU_PATH` | `"rockyou.txt"` | Path to the dataset file |
| `SHAP_SAMPLE` | `1_000` | Number of samples for SHAP (higher = slower) |
| `PERM_REPEATS` | `5` | Repeats per feature for permutation importance |
| `OUTPUT_DIR` | `"output"` | Folder where all PNGs are saved |

---

## 9. Deactivate the Virtual Environment

When you're done:

```bash
deactivate
```