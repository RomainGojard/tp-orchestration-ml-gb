from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_yolov8

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_yolov8,
            inputs=dict(
                model_path="params:trained_model_path",  # un fichier .pt
                data_yaml_path="params:data_yaml_path"  # un fichier YAML
            ),
            outputs="params:yolo_evaluation_metrics",
            name="evaluate_yolov8_node",
        ),
    ])
