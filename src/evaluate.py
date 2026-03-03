from sklearn.metrics import mean_absolute_error

def evaluate(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)