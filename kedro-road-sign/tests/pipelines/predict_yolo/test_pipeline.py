from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import cv2
import numpy as np

from src.kedro_road_sign.pipelines.predict_yolo.nodes import predict_yolov8

@patch("src.kedro_road_sign.pipelines.predict_yolo.nodes.predict_yolov8")
@patch("cv2.imread")
def test_predict_yolov8_no_predictions_creates_no_file(mock_imread, mock_yolo):
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_image

    # Mock : aucune boîte prédite
    mock_prediction = [MagicMock()]
    mock_prediction[0].boxes = []

    mock_model = MagicMock()
    mock_model.return_value = mock_prediction
    mock_yolo.return_value = mock_model

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        images_input_dir = tmpdir_path / "images"
        output_dir = tmpdir_path / "output"
        images_input_dir.mkdir()
        output_dir.mkdir()

        image_path = images_input_dir / "test.png"
        cv2.imwrite(str(image_path), dummy_image)

        data_config = {"names": ["stop", "yield"]}

        predict_yolov8("data/06_models/best.pt", str(images_input_dir), data_config, str(output_dir))

        txt_file = output_dir / "test.txt"
        assert not txt_file.exists(), "Un fichier ne devrait pas être créé sans prédictions"

@patch("src.kedro_road_sign.pipelines.predict_yolo_api.nodes.YOLO")
@patch("cv2.imread")
def test_predict_yolov8_with_prediction(mock_imread, mock_yolo):
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_image

    # Mock d'une boîte prédite
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.xywh = [50, 50, 20, 20]
    mock_prediction = [MagicMock()]
    mock_prediction[0].boxes = [mock_box]

    mock_model = MagicMock()
    mock_model.return_value = mock_prediction
    mock_yolo.return_value = mock_model

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        images_input_dir = tmpdir_path / "images"
        output_dir = tmpdir_path / "output"
        images_input_dir.mkdir()
        output_dir.mkdir()

        image_path = images_input_dir / "test.png"
        cv2.imwrite(str(image_path), dummy_image)

        data_config = {"names": ["stop", "yield"]}

        from src.kedro_road_sign.pipelines.predict_yolo_api.nodes import predict_yolov8
        predict_yolov8("data/06_models/best.pt", str(images_input_dir), data_config, str(output_dir))

        txt_file = output_dir / "test.txt"
        assert txt_file.exists()
        content = txt_file.read_text()
        assert content.startswith("0 ")
