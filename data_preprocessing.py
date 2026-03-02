import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
X_full = pd.read_csv(
    r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\concrete.csv"
)

# Separate target
target_col = "CompressiveStrength"
X_full.dropna(subset=[target_col], axis=0, inplace=True)
y = X_full.pop(target_col)

# Identify feature columns
numerical_cols = [col for col in X_full.columns if X_full[col].dtype in ["int64", "float64"]]
categorical_cols = [col for col in X_full.columns if X_full[col].dtype == "object"]

feature_cols = numerical_cols + categorical_cols
X = X_full[feature_cols]

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline with placeholder model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=0))
])

# Hyperparameter grid
param_grid = {
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [10, 20, 40],
    'regressor__min_samples_split': [2, 5, 10]
}

# Grid search
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-4
)

grid.fit(X_train, y_train)

# Best model and evaluation
best_model = grid.best_estimator_
preds = best_model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)

print(f"Best parameters:=", grid.best_params_)
print(f"Best CV MAE:", -grid.best_score_)
print(f"Validation MAE:", mae)