from src.data_loader import load_data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.preprocessing import build_preprocessor

# -----------------------------
# Load data
# -----------------------------
X, y = load_data(r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\raw\diamonds.csv"
                  ,target_col="price"
                #   ,index_column="name"
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

# Fit the preprocessor on training data only
preprocessor.fit(X_train)

# Transform both train and test sets
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)


# -----------------------------
# XGBoost model with early stopping
# -----------------------------
xgb = XGBRegressor(
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

xgb.fit(
    X_train_t,
    y_train,
    eval_set=[(X_test_t, y_test)],
    verbose=False
)


# -----------------------------
# Evaluation
# -----------------------------
preds = xgb.predict(X_test_t)

mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:.3g}")
print(f"RMSE: {rmse:.3g}")
print(f"R²:   {r2:.4f}")

# -----------------------------
# SHAP value analysis
# -----------------------------
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test_t)

# Beeswarm summary plot
shap.summary_plot(
    shap_values.values,
    X_test_t,
    feature_names=preprocessor.get_feature_names_out()
)

# Bar plot of global importance
shap.summary_plot(
    shap_values.values,
    X_test_t,
    feature_names=preprocessor.get_feature_names_out(),
    plot_type="bar"
)

# -----------------------------
# Permutation importance
# -----------------------------
from sklearn.inspection import permutation_importance

# Compute permutation importance on the transformed test set
perm = permutation_importance(
    xgb,
    X_test_t,
    y_test,
    n_repeats=10,
    random_state=0,
    scoring="neg_mean_absolute_error"
)

# Extract feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Sort by importance
sorted_idx = perm.importances_mean.argsort()[::-1] # type: ignore

print("\nPermutation Importance (top 20 features):")
for idx in sorted_idx[:20]:
    print(f"{feature_names[idx]}: {perm.importances_mean[idx]:.4f}") # type: ignore