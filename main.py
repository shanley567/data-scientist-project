from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.models import get_random_forest, param_grid_rf
from src.train import train_model
from src.evaluate import evaluate
from sklearn.model_selection import train_test_split

X, y = load_data(r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\raw\concrete.csv",
                  "CompressiveStrength")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

preprocessor = build_preprocessor(X_train)
model = get_random_forest()

grid = train_model(preprocessor, model, param_grid_rf, X_train, y_train)

mae = evaluate(model=grid.best_estimator_, X_valid=X_valid, y_valid=y_valid)
print(f"Validation MAE: {mae:.3g} (3sf)")