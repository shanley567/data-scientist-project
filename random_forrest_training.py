from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.models import get_random_forest, param_grid_rf
from src.train import train_model
from src.evaluate import evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# -----------------------------
# Load data
# -----------------------------
X, y = load_data(
    r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\raw\diamonds.csv",
    target_col="price"
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
# Predict on RAW validation data
# (pipeline handles preprocessing)
# -----------------------------
preds = best_model.predict(X_valid)

# -----------------------------
# Evaluation metrics
# -----------------------------
mae = mean_absolute_error(y_valid, preds)
rmse = root_mean_squared_error(y_valid, preds)
r2 = r2_score(y_valid, preds)

print("\nFinal Evaluation on Validation Set:")
print(f"MAE:  {mae:.3g}")
print(f"RMSE: {rmse:.3g}")
print(f"R²:   {r2:.4f}")