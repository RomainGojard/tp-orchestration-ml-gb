from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_ocr_data, ocr, configure_tesseract

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_ocr_data,
            inputs=dict(
                labels_path="params:labels_input",
                images_path="params:image_api_path",
                images_path_preprocessed="params:images_path_preprocessed",
                data_config="data_config"
            ),
            outputs="ocr_rois",
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
            func=ocr,
            inputs=dict(
                rois="ocr_rois",
                lang="params:ocr_lang"
            ),
            outputs="ocr_metrics",
            name="ocr"
        )
    ])
