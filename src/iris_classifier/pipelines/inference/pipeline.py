from kedro.pipeline import Pipeline, node
from .nodes import run_inference

def create_pipeline(**kwargs):
    return Pipeline([
        node(run_inference, ["model", "X_test"], "y_pred", name="run_inference"),
    ])