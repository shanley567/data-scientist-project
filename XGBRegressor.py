from src.data_loader import load_data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.preprocessing import build_preprocessor
from src.evaluate import evaluate

from src.explain.shap_explainer import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar
)
from src.explain.permutation_importance import compute_permutation_importance


# -----------------------------
# Load data
# -----------------------------
X, y = load_data(
    r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\raw\diamonds.csv"
    , target_col="price"
    # , index_column="id"
)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = build_preprocessor(X_train)

# Fit on training data only
preprocessor.fit(X_train)

# Transform train and test sets
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

# -----------------------------
# XGBoost model with early stopping
# -----------------------------
model = XGBRegressor(
    random_state=0,
    n_estimators=5000,
    early_stopping_rounds=50,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)

model.fit(
    X_train_t,
    y_train,
    eval_set=[(X_test_t, y_test)],
    verbose=False
)

# -----------------------------
# Evaluation (using shared evaluator)
# -----------------------------
metrics = evaluate(model=model, X=X_test_t, y=y_test)

print("\nXGBRegressor - Final Evaluation on Test Set:")
print(f"MAE:  {metrics['mae']:.3g}")
print(f"RMSE: {metrics['rmse']:.3g}")
print(f"R²:   {metrics['r2']:.4f}")

# -----------------------------
# SHAP (modular)
# -----------------------------
feature_names = preprocessor.get_feature_names_out()

shap_values, X_dense = compute_shap_values(
    model=model,
    X=X_test_t,
    feature_names=feature_names,
    max_samples=500
)

plot_shap_summary(shap_values, X_dense, feature_names)
plot_shap_bar(shap_values, X_dense, feature_names)

# -----------------------------
# Permutation importance (modular)
# -----------------------------
importances = compute_permutation_importance(
    model=model,
    X=X_test_t,
    y=y_test,
    feature_names=feature_names
)

print("\nPermutation Importance (top 20 features):")
for name, score in importances[:20]:
    print(f"{name}: {score:.4f}")
