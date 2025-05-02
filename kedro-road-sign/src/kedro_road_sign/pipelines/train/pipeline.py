# src/my_project/pipelines/yolo_pipeline/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolov8  # ❌ enlever preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_yolov8,
            inputs="params:data_config_path",
            outputs="training_output",
            name="train_yolov8_node",
        ),
    ])
