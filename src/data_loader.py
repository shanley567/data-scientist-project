import pandas as pd

def load_concrete_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["CompressiveStrength"])
    y = df.pop("CompressiveStrength")
    return df, y