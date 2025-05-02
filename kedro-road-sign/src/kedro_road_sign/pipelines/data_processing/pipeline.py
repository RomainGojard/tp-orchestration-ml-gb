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
                inputs=dict(data="raw_train_data", output_dir="yolo_output_dir"),
                outputs=None,
                name="preprocess_train_data_node",
            ),
        ]
    )
