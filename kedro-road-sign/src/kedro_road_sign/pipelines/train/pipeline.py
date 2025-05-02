from kedro.pipeline import Pipeline, node
from .nodes import train_yolo_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_yolo_model,
                inputs=dict(
                    data_dir="params:yolo_data_dir",
                    model_config="params:yolo_model_config",
                    epochs="params:yolo_epochs",
                    output_dir="params:trained_yolo_model_path",
                ),
                outputs="trained_yolo_model_path",
                name="train_yolo_model_node",
            ),
        ]
    )