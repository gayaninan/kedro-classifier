from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_model,
            inputs=["X_train", "y_train", "parameters"],
            outputs="model",
            name="train_model"
        )
    ])