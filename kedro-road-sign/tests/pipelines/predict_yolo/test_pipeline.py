"""
This is a boilerplate test file for pipeline 'predict_yolo'
generated using Kedro 0.19.13.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
from src.kedro_road_sign.pipelines.predict_yolo.nodes import predict_yolov8

def test_predict_yolov8():
  predict_yolov8(model_path: str, images_yolo_predict_input_path: str, data_config: dict, yolo_predict_output_dir: str)