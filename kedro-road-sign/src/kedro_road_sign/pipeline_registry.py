"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro_road_sign.pipelines.train import pipeline as train_pipeline
from kedro_road_sign.pipelines.data_processing import pipeline as data_processing_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "train": train_pipeline.create_pipeline(),
        "data_processing": data_processing_pipeline.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines