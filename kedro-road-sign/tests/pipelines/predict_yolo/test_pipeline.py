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

import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import numpy as np
import cv2

@patch("nodes.YOLO")
@patch("cv2.imread")
def test_predict_yolov8_creates_txt_file(self, mock_imread, mock_yolo):
  # Mock image and prediction
  mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

  mock_box = MagicMock()
  mock_box.cls = [0]
  mock_box.xywh = [50, 50, 20, 20]

  mock_result = MagicMock()
  mock_result.boxes = [mock_box]
  mock_yolo_instance = MagicMock(return_value=[mock_result])
  mock_yolo.return_value = mock_yolo_instance

  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    input_dir = tmpdir / "input"
    output_dir = tmpdir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Fake input image
    img_path = input_dir / "test.png"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    predict_yolov8(
      model_path="dummy_model.pt",
      images_yolo_predict_input_path=str(input_dir),
      data_config={"names": ["class0"]},
      yolo_predict_output_dir=str(output_dir)
    )

    txt_file = output_dir / "test.txt"
    self.assertTrue(txt_file.exists(), "Le fichier .txt de prédiction n'a pas été créé.")
    content = txt_file.read_text().strip()
    self.assertRegex(content, r"^0 0\.\d+ 0\.\d+ 0\.\d+ 0\.\d+$")

