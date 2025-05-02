"""
This is a boilerplate pipeline 'import_data'
generated using Kedro 0.19.11
"""

from kedro.pipeline import Pipeline, node
from .nodes import import_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=import_data,
                inputs=dict(
                    dataset_name="params:dataset_name",
                    download_path="params:download_path"
                ),
                outputs="raw_data_path",
                name="import_data_node",
            ),
        ]
    )
