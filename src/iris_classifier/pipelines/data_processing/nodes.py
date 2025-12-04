import pandas as pd
from typing import Dict, Tuple

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_split = parameters["train_split"]
    target_column = parameters["target_column"]

    data_train = data.sample(frac=train_split, random_state=parameters.get("random_state", 42))
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=[target_column])
    X_test = data_test.drop(columns=[target_column])
    y_train = pd.DataFrame(data_train[target_column], columns=[target_column])
    y_test = pd.DataFrame(data_test[target_column], columns=[target_column])

    return X_train, X_test, y_train, y_test