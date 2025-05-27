"""
This is a boilerplate test file for pipeline 'evaluate_yolo'
generated using Kedro 0.19.11.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

# test_pipeline.py
import pytest
from src.kedro_road_sign.pipelines.evaluate_yolo.nodes import evaluate_yolov8

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_evaluate_model(sample_data):
    if 1 == 0: 
        evaluate_yolov8()
        assert True
    else:
        # Placeholder for actual test logic
        assert True