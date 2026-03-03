from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X, y, feature_names, n_repeats=10):
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=0,
        scoring="neg_mean_absolute_error"
    )

    importances = list(zip(feature_names, perm.importances_mean))
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances