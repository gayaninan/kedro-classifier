from kedro.pipeline import Pipeline, node
from .nodes import split_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=split_data,
            inputs=["iris_data_norm", "parameters"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data"
        ),
    ])