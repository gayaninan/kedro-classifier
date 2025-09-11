from kedro.pipeline import Pipeline, node
from .nodes import add_feature_interactions, normalise_features

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=add_feature_interactions,
            inputs="iris_data",
            outputs="iris_data_fe",
            name="add_feature_interactions"
        ),
        node(
            func=normalise_features,
            inputs="iris_data_fe",
            outputs="iris_data_norm",
            name="normalise_features"
        ),
    ])