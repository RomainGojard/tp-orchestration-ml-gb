"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro_road_sign.pipelines.train import pipeline as train_pipeline
from kedro_road_sign.pipelines.data_processing import pipeline as data_processing_pipeline
from kedro_road_sign.pipelines.evaluate_yolo import pipeline as evaluate_pipeline
from kedro_road_sign.pipelines.OCR import pipeline as ocr_pipeline
from kedro_road_sign.pipelines.predict_yolo import pipeline as predict_yolo_pipeline
from kedro_road_sign.pipelines.write_label_files import pipeline as write_label_files_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_processing": data_processing_pipeline.create_pipeline(),
        "train": train_pipeline.create_pipeline(),
        "OCR": ocr_pipeline.create_pipeline(),
        "evaluate_yolo": evaluate_pipeline.create_pipeline(),
        "predict_yolo": predict_yolo_pipeline.create_pipeline(),
        "write_label_files": write_label_files_pipeline.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["model_training"] = pipelines["train"] + pipelines["evaluate_yolo"]

    return pipelines