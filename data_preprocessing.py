import pandas as pd

X_full = pd.read_csv(r"C:\Users\js105\Documents\Coding_portfolio\data-scientist-project\data\concrete.csv")

# seperate target from data
X = X_full.copy()

target_col = "CompressiveStrength"
try:

    X.dropna(subset=[target_col] # Drops rows with missing target values
            , axis=0 # Drop rows (not columns)
            , inplace=True) # Changes X directly and not a copy

    y = X.pop(target_col) # Removes target column from X and stores it in y

except KeyError:
    print(f"Error: Target column NOT found '{target_col}'. \nPossible columns are: {X.columns.tolist()}")


# Defining categorical and numerical columns for feature cols
numerical_cols = [col for col in X.columns 
                    if X[col].dtype in ["int64", "float64"]]

categorical_cols = [col for col in X.columns 
                    if X[col].dtype == "object"]

feature_cols = numerical_cols + categorical_cols # Merging col names 
X = X[feature_cols] # Make sure X only contains the feature columns

