from kedro.pipeline import Pipeline, node
from .nodes import retrain_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=retrain_model,
            inputs=["X_train", "y_train", "parameters"],
            outputs="retrained_model",
            name="retrain_model"
        ),
    ])