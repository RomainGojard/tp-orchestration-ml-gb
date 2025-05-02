from kedro.pipeline import Pipeline, node, pipeline
from .nodes import process_labels
from pathlib import Path

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_labels,
            inputs=dict(
                csv_path="train_csv",
                image_base_path="raw_data_dir",
                output_dir="processed_data_dir",
                subset="params:train_subset"
            ),
            outputs=None,
            name="process_train_labels"
        ),
        node(
            func=process_labels,
            inputs=dict(
                csv_path="test_csv",
                image_base_path="raw_data_dir",
                output_dir="processed_data_dir",
                subset="params:test_subset"
            ),
            outputs=None,
            name="process_test_labels"
        )
    ])
