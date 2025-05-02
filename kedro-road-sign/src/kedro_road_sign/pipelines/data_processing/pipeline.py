from kedro.pipeline import Pipeline, node
from .nodes import load_data, preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="params:train_csv_path",
                outputs="raw_train_data",
                name="load_train_data_node",
            ),
            node(
                func=preprocess_data,
                inputs="raw_train_data",
                outputs="yolo_output_dir",
                name="preprocess_train_data_node",
            ),
        ]
    )
