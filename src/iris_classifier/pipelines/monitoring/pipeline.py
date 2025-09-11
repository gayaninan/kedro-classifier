from kedro.pipeline import Pipeline, node
from .nodes import monitor_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=monitor_data,
            inputs=["X_train", "X_test"],
            outputs="drift_report",
            name="monitor_data"
        ),
    ])