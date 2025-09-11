import pandas as pd

def add_feature_interactions(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns
    if len(cols) >= 2:
        data["feature_interaction"] = data[cols[0]] * data[cols[1]]
    return data

def normalise_features(data: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include="number").columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
    return data