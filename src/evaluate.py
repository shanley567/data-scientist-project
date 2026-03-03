from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.utils import load_model

def evaluate(model=None, X=None, y=None, load_saved=False):
    if load_saved:
        model = load_model()

    if model is None:
        raise ValueError("No model provided. Set load_saved=True or pass a trained model.")

    if X is None or y is None:
        raise ValueError("X and y must be provided for evaluation.")

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }