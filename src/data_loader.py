import pandas as pd

def load_data(path, target_col, index_column=None):
    # Try loading the CSV
    try:
        df = pd.read_csv(path, index_col=index_column)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File not found. Place the CSV file into the 'data/' directory."
        )
    except ValueError as e:
        # This happens when index_column is invalid
        raise ValueError(
            f"Invalid index column '{index_column}'."
            f"Available columns: {pd.read_csv(path).columns.tolist()}"
        ) from e

    # Validate target column
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[target_col])
    y = df.pop(target_col)
    return df, y