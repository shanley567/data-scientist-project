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
    print(f"Error: Target column NOT found '{target_col}' . \nPossible columns are: {X.columns.tolist()}")
