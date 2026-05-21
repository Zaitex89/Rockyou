import os
from config   import OUTPUT_DIR
from data     import load_passwords, run_eda
from features import build_dataframe
from model    import prepare_data, tune, train, evaluate, feature_importance, run_permutation, run_shap

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Steps 1-2: load and EDA
    passwords = load_passwords()
    run_eda(passwords)

    # Steps 3-4: feature extraction and pattern classification
    df, pattern_counts = build_dataframe(passwords)

    # Steps 5-6: hyperparameter tuning and training
    X_train, X_test, y_train, y_test, feature_cols, le = prepare_data(df)
    best  = tune(X_train, X_test, y_train, y_test, le)
    model = train(X_train, X_test, y_train, y_test, best, le)

    # Steps 7-10: evaluation and analysis
    evaluate(model, X_test, y_test, le)
    importances = feature_importance(model, feature_cols, pattern_counts)
    run_permutation(model, X_test, y_test, feature_cols, importances)
    run_shap(model, X_test)

    print(f"\n  Sample size : {len(passwords):,}")
    print(f"  Patterns    : {len(le.classes_)}")
    print(f"  Best params : max_depth={int(best.max_depth)}, lr={best.learning_rate}")

if __name__ == "__main__":
    main()