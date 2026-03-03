import shap
import numpy as np
from scipy.sparse import issparse

def compute_shap_values(model, X, feature_names, max_samples=None):
    if max_samples:
        X = X[:max_samples]

    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_dense)

    return shap_values, X_dense


def plot_shap_summary(shap_values, X_dense, feature_names):
    shap.summary_plot(
        shap_values.values,
        X_dense,
        feature_names=feature_names
    )


def plot_shap_bar(shap_values, X_dense, feature_names):
    shap.summary_plot(
        shap_values.values,
        X_dense,
        feature_names=feature_names,
        plot_type="bar"
    )