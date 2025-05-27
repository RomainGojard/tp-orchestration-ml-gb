"""
This is a boilerplate test file for pipeline 'OCR'
generated using Kedro 0.19.11.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
import numpy as np
import cv2
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.kedro_road_sign.pipelines.OCR.nodes import (
    prepare_ocr_data, evaluate_ocr, compute_cer, configure_tesseract
)


@pytest.fixture
def dummy_image(tmp_path):
    """Crée une image temporaire simulée pour les tests"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image_path = tmp_path / "image1.png"
    cv2.imwrite(str(image_path), img)
    return image_path


@pytest.fixture
def dummy_label_file(tmp_path):
    """Crée un fichier de label temporaire avec des coordonnées normalisées"""
    label_path = tmp_path / "image1.txt"
    label_content = "0 0.5 0.5 0.2 0.2"  # class_id x y w h
    label_path.write_text(label_content)
    return label_path


@pytest.fixture
def dummy_data_config():
    return {
        "names": ["stop", "yield", "50_speed"]
    }


@pytest.fixture
def dummy_preprocessed_dir(tmp_path):
    return tmp_path / "preprocessed"


def test_prepare_ocr_data_success(dummy_image, dummy_label_file, dummy_data_config, dummy_preprocessed_dir, tmp_path):
    # Arrange
    images_path = tmp_path
    labels_path = tmp_path
    preprocessed_path = dummy_preprocessed_dir
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    # Patch les fonctions de traitement d'image
    with patch("kedro_road_sign.pipelines.OCR.ocr_pipeline.remove_noise", side_effect=lambda x: x), \
         patch("kedro_road_sign.pipelines.OCR.ocr_pipeline.grayscale", side_effect=lambda x: x), \
         patch("kedro_road_sign.pipelines.OCR.ocr_pipeline.opening", side_effect=lambda x: x):

        # Act
        result = prepare_ocr_data(str(images_path), str(labels_path), dummy_data_config, str(preprocessed_path))

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "stop"
        assert Path(result[0]["image"]).exists()
        assert isinstance(result[0]["roi"], tuple)


def test_prepare_ocr_data_raises_on_missing_image(tmp_path, dummy_label_file, dummy_data_config):
    with pytest.raises(FileNotFoundError):
        prepare_ocr_data(
            images_path=str(tmp_path),
            labels_path=str(tmp_path),
            data_config=dummy_data_config,
            images_path_preprocessed=str(tmp_path / "out")
        )


def test_compute_cer_typical():
    ref = "STOP"
    hyp = "ST0P"  # 1 substitution
    cer = compute_cer(ref, hyp)
    assert cer == 1
