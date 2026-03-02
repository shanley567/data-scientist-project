import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

X_full = pd.read_csv(r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\concrete.csv")

  # seperate target from data
X = X_full.copy()

target_col = "CompressiveStrength"
try: 

    X.dropna(subset=[target_col] # Drops rows with missing target values
            , axis    = 0     # Drop rows (not columns)
            , inplace = True  # Changes X directly and not a copy
    )

    y = X.pop(target_col)  # Removes target column from X and stores it in y

except KeyError: 
    print(f"Error: Target column NOT found '{target_col}'. \nPossible columns are: {X.columns.tolist()}")


  # Defining categorical and numerical columns for feature cols
numerical_cols = [col for col in X.columns 
                    if X[col].dtype in ["int64", "float64"]]

categorical_cols = [col for col in X.columns 
                    if X[col].dtype == "object"]

feature_cols = numerical_cols + categorical_cols    # Merging col names 
X         = X[feature_cols]  # Make sure X only contains the feature columns

# Splitting of data into training and test sets
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                      y, 
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)


  # Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='median')),  # Imputes null values = median
    ('scaler', StandardScaler())                                    # Scales num values to mean=0, stand_dev=1 - used mainly for NNs but does not affect linear regressions 
])


categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fills null vals with most frequent vals
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Encodes cardinality columns, ignore unknown categories in test data that were not present in training data
])

  # Batches the transformers and tells to apply to correct columns
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
