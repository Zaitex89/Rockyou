OUTPUT_DIR   = "output"
SAMPLE_SIZE  = 100_000
ROCKYOU_PATH = "rockyou.txt"
SHAP_SAMPLE  = 1_000
PERM_REPEATS = 5

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

PARAM_GRID = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}