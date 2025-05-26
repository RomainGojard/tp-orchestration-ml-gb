from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_ocr_data, evaluate_ocr, configure_tesseract

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_ocr_data,
            inputs=dict(
                labels_path="params:labels_path",
                images_path="params:images_path",
                data_config_path="params:data_config_path"
            ),
            outputs="params:ocr_rois",
            name="prepare_ocr_data_node"
        ),
        node(
            func=configure_tesseract,
            inputs=dict(
                path_cmd="params:cmd_local_path"
            ),
            outputs=None,
            name="configure_tesseract_node"
        ),
        node(
            func=evaluate_ocr,
            inputs=dict(
                rois="params:ocr_rois",
                lang="params:ocr_lang"
            ),
            outputs="ocr_metrics",
            name="evaluate_ocr_node"
        )
    ])
