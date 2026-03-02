import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_preprocess(train_path, test_path=None, target_col="SalePrice"):
    # Load training data
    X_full = pd.read_csv(train_path, index_col='Id')

      # Separate target
    X = X_full.copy()
    X.dropna(subset=[target_col], axis=0, inplace=True)
    y = X.pop(target_col)


    # Select useful columns
    num_cols = [c for c in X.columns 
                if X[c].dtype in ['float64', 'int64']]  # Select numerical columns
    cat_cols = [c for c in X.columns 
                if X[c].dtype == 'object' and X[c].nunique() < 10]  # Select categorical columns with low cardinality
    useful_cols = num_cols + cat_cols
    X = X[useful_cols]
    
    # Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Imputes null values = 0
        ('scaler', StandardScaler())                                    # Scales num values to mean=0, stand_dev=1 - used mainly for NNs but does not affect linear regressions 
    ])


    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fills null vals with most frequent vals
            ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Encodes cardinality columns 
    ])

    # Batches the transformers and tells to apply to correct cols
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    # If test_path is provided, process test data too
    if test_path is not None:
        X_test_full = pd.read_csv(test_path, index_col='Id')
        X_test = X_test_full[useful_cols].copy()
        return X, y, X_test, preprocessor
    else:
        return X, y, None, preprocessor
    
    # Note for the returned values
    # X - the raw data with all columns except the target, not the preprocessed data
    # y - the target row
    # X_test - if available returns raw data
    # preprocessor - returns the function that has not yet been activated using the transform function