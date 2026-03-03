from sklearn.metrics import mean_absolute_error
from src.utils import load_model

def evaluate(model=None, X_valid=None, y_valid=None, load_saved=False):
    if load_saved:
        model = load_model()

    if model is None:
        raise ValueError("No model provided. Set load_saved=True or pass a trained model.")

    if X_valid is None or y_valid is None:
        raise ValueError("X_valid and y_valid must be provided for evaluation.")

    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)