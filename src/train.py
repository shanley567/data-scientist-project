from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.utils import save_model

def train_model(preprocessor, model, param_grid, X_train, y_train):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-3
    )

    grid.fit(X_train, y_train)

    # Save the best model
    save_model(grid.best_estimator_)

    return grid