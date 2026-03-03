import joblib
from pathlib import Path

def save_model(model, filename="concrete_strength_model.joblib"):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    path = models_dir / filename
    joblib.dump(model, path)
    return path

def load_model(filename="concrete_strength_model.joblib"):
    path = Path("models") / filename
    return joblib.load(path)