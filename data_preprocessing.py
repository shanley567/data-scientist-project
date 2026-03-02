import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
X_full = pd.read_csv(r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\concrete.csv")

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

# Full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=0))
])

# Fit model
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Validation MAE:", mae)