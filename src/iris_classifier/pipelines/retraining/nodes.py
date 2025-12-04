from iris_classifier.pipelines.training.nodes import train_model

def retrain_model(X_train, y_train, parameters):
    # Just call train_model again
    return train_model(X_train, y_train, parameters)