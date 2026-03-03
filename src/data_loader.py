import pandas as pd

def load_data(path, target_col):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File not found. Place csv data file into 'data/' directory"
        )


    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[target_col])
    y = df.pop(target_col)
    return df, y