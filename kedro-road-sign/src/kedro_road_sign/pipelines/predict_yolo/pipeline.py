from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict_yolov8

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict_yolov8,
            inputs=["params:model_path", "params:images_path", "data_config", "params:yolo_predict_output_dir"],
            outputs= None,
            name="predict_yolov8_node"
        ),
    ])
