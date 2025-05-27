"""
This is a boilerplate pipeline 'write_label_files'
generated using Kedro 0.19.13
"""
from .nodes import empty_input_model_folder, copy_files
from kedro.pipeline import node, Pipeline, pipeline  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=empty_input_model_folder,
            inputs = "params:labels_input",
            outputs="empty_flag",
            name="write_files_node"
        ),
        node(
            func=copy_files,
            inputs = dict(
                source_dir="params:yolo_predict_output_dir", 
                dest_dir="params:labels_input",
                 _trigger="empty_flag"
            ),
            outputs=None,
            name="copy_predicts_results"
        ),
        node(
            func=copy_files,
            inputs = dict(
                source_dir="params:labels_use_case_path",
                dest_dir="params:labels_input",
                 _trigger="empty_flag"
            ),
            outputs=None,
            name="copy_use_cases"
        ),
    ])