from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.models import get_random_forest, param_grid_rf
from src.train import train_model
from src.evaluate import evaluate

from sklearn.model_selection import train_test_split

# NEW imports
from src.explain.shap_explainer import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar
)
from src.explain.permutation_importance import (
    compute_permutation_importance
)

# -----------------------------
# Load data
# -----------------------------
X, y = load_data(
    r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\raw\diamonds.csv"
    ,target_col="price"
    # ,index_column=None
)

# -----------------------------
# Train/validation split
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------
# Preprocessor
# -----------------------------
preprocessor = build_preprocessor(X_train)

# -----------------------------
# Random Forest model + GridSearchCV
# -----------------------------
model = get_random_forest()

grid = train_model(
    preprocessor=preprocessor,
    model=model,
    param_grid=param_grid_rf,
    X_train=X_train,
    y_train=y_train
)

print(f"Best parameters: {grid.best_params_}")

# -----------------------------
# Use the trained pipeline
# -----------------------------
best_model = grid.best_estimator_

# -----------------------------
# Evaluate on RAW validation data
# -----------------------------
metrics = evaluate(model=best_model, X=X_valid, y=y_valid)

print("\nRandom Forest - Final Evaluation on Validation Set:")
print(f"MAE:  {metrics['mae']:.3g}")
print(f"RMSE: {metrics['rmse']:.3g}")
print(f"R²:   {metrics['r2']:.3g}")

# # -----------------------------
# # SHAP explainability
# # -----------------------------
# pre = best_model.named_steps["preprocessor"]
# rf = best_model.named_steps["regressor"]

# X_valid_t = pre.transform(X_valid)
# feature_names = pre.get_feature_names_out()

# shap_values, X_dense = compute_shap_values(
#     model=rf,
#     X=X_valid_t,
#     feature_names=feature_names,
#     max_samples=500
# )

# plot_shap_summary(shap_values, X_dense, feature_names)
# plot_shap_bar(shap_values, X_dense, feature_names)

# # -----------------------------
# # Permutation importance
# # -----------------------------
# importances = compute_permutation_importance(
#     model=rf,
#     X=X_valid_t,
#     y=y_valid,
#     feature_names=feature_names
# )

# print("\nPermutation Importance (top 20 features):")
# for name, score in importances[:20]:
#     print(f"{name}: {score:.4f}")