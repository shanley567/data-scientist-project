from src.data_loader import load_data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.preprocessing import build_preprocessor
from src.evaluate import evaluate   # <-- use your modular evaluator

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

# # -----------------------------
# # SHAP value analysis
# # -----------------------------
# import shap

# explainer = shap.TreeExplainer(model)
# shap_values = explainer(X_test_t)

# shap.summary_plot(
#     shap_values.values,
#     X_test_t,
#     feature_names=preprocessor.get_feature_names_out()
# )

# shap.summary_plot(
#     shap_values.values,
#     X_test_t,
#     feature_names=preprocessor.get_feature_names_out(),
#     plot_type="bar"
# )

# # -----------------------------
# # Permutation importance
# # -----------------------------
# from sklearn.inspection import permutation_importance

# perm = permutation_importance(
#     model,
#     X_test_t,
#     y_test,
#     n_repeats=10,
#     random_state=0,
#     scoring="neg_mean_absolute_error"
# )

# feature_names = preprocessor.get_feature_names_out()
# sorted_idx = perm.importances_mean.argsort()[::-1] 

# print("\nPermutation Importance (top 20 features):")
# for idx in sorted_idx[:20]:
#     print(f"{feature_names[idx]}: {perm.importances_mean[idx]:.4f}")