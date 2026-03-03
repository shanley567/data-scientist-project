import pandas as pd

def load_data(path, target_col):
    df = pd.read_csv(path)
    df = df.dropna(subset=[target_col])
    y = df.pop(target_col)
    return df, y