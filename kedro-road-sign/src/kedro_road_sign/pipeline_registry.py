"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro_road_sign.pipelines.train import pipeline as train_pipeline
from kedro_road_sign.pipelines.evaluate_yolo import pipeline as evaluate_pipeline
from kedro_road_sign.pipelines.OCR import pipeline as ocr_pipeline
from kedro_road_sign.pipelines.predict_yolo import pipeline as predict_yolo_pipeline
from kedro_road_sign.pipelines.write_label_files import pipeline as write_label_files_pipeline
from kedro_road_sign.pipelines.predict_yolo_api import pipeline as predict_yolo_api_pipeline
from kedro_road_sign.pipelines.OCR_api import pipeline as ocr_api_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "train": train_pipeline.create_pipeline(),
        "OCR": ocr_pipeline.create_pipeline(),
        "OCR_api": ocr_api_pipeline.create_pipeline(),  # Alias for OCR
        "evaluate_yolo": evaluate_pipeline.create_pipeline(),
        "predict_yolo": predict_yolo_pipeline.create_pipeline(),
        "predict_yolo_api": predict_yolo_api_pipeline.create_pipeline(),  # Alias for predict_yolo
        "write_label_files": write_label_files_pipeline.create_pipeline(),
    }
    pipelines["use_cases"] = pipelines["predict_yolo"] + pipelines["write_label_files"] + pipelines["OCR"]
    pipelines["model_training"] = pipelines["train"] + pipelines["evaluate_yolo"]
    pipelines["prediction"] = pipelines["predict_yolo"] + pipelines["write_label_files"] + pipelines["OCR_api"]

    return pipelines