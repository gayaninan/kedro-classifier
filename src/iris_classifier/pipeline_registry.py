from iris_classifier.pipelines.feature_engineering import pipeline as feature_engineering_pipeline
from iris_classifier.pipelines.data_processing import pipeline as data_processing_pipeline
from iris_classifier.pipelines.training import pipeline as training_pipeline
from iris_classifier.pipelines.inference import pipeline as inference_pipeline
from iris_classifier.pipelines.retraining import pipeline as retraining_pipeline
from iris_classifier.pipelines.monitoring import pipeline as monitoring_pipeline

def register_pipelines():
    return {
        "data_processing": data_processing_pipeline.create_pipeline(),
        "feature_engineering": feature_engineering_pipeline.create_pipeline(),
        "training": training_pipeline.create_pipeline(),
        "inference": inference_pipeline.create_pipeline(),
        "retraining": retraining_pipeline.create_pipeline(),
        "monitoring": monitoring_pipeline.create_pipeline(),
        "__default__": (
            feature_engineering_pipeline.create_pipeline()
            + data_processing_pipeline.create_pipeline()
            + training_pipeline.create_pipeline()
            + inference_pipeline.create_pipeline()
            + monitoring_pipeline.create_pipeline()
        ),
    }


        