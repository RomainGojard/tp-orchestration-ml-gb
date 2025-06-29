import pytest
import numpy as np
import cv2
from pathlib import Path
from src.kedro_road_sign.pipelines.OCR_api.nodes import *
def test_grayscale():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    gray = grayscale(img)
    assert len(gray.shape) == 2
    assert gray.shape == (10, 10)

def test_remove_noise():
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    denoised = remove_noise(img)
    assert denoised.shape == img.shape

def test_opening():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    opened = opening(img)
    assert opened.shape == img.shape

def test_thresholding():
    img = np.ones((10, 10), dtype=np.uint8) * 127
    thresh = thresholding(img)
    assert thresh.shape == img.shape
    assert np.all((thresh == 0) | (thresh == 255))

def test_dilate_erode():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 255
    dilated = dilate(img)
    eroded = erode(img)
    assert dilated.shape == img.shape
    assert eroded.shape == img.shape

def test_canny():
    img = np.ones((10, 10), dtype=np.uint8) * 255
    edges = canny(img)
    assert edges.shape == img.shape

def test_deskew():
    img = np.ones((10, 10), dtype=np.uint8) * 255
    rotated = deskew(img)
    assert rotated.shape == img.shape

def test_match_template():
    img = np.ones((10, 10), dtype=np.uint8) * 255
    template = np.ones((3, 3), dtype=np.uint8) * 255
    result = match_template(img, template)
    assert result.shape[0] > 0

def test_compute_cer():
    ref = "hello"
    hyp = "h3llo"
    cer = compute_cer(ref, hyp)
    assert isinstance(cer, int)
    assert cer >= 0

def test_configure_tesseract(monkeypatch):
    path = "/usr/bin/tesseract"
    configure_tesseract(path)
    assert hasattr(pytesseract.pytesseract, "tesseract_cmd")
    assert pytesseract.pytesseract.tesseract_cmd == path

# Pour prepare_ocr_data et ocr, il faut des fichiers images et labels factices.
# Exemple de test minimaliste (à adapter selon tes fixtures/données de test) :

def test_prepare_ocr_data(tmp_path):
    # Crée un faux fichier image et label
    img_path = tmp_path / "images"
    label_path = tmp_path / "labels"
    preproc_path = tmp_path / "preproc"
    img_path.mkdir()
    label_path.mkdir()
    preproc_path.mkdir()
    # Image factice
    img_file = img_path / "test.png"
    cv2.imwrite(str(img_file), np.ones((10, 10, 3), dtype=np.uint8) * 255)
    # Label factice (class_id, x, y, w, h)
    label_file = label_path / "test.txt"
    label_file.write_text("0 0.5 0.5 0.5 0.5")
    # Data config factice
    data_config = {"names": ["test_label"]}
    rois = prepare_ocr_data(
        images_path=str(img_path),
        labels_path=str(label_path),
        data_config=data_config,
        images_path_preprocessed=str(preproc_path)
    )
    assert isinstance(rois, list)
    assert len(rois) == 1
    assert rois[0]["label"] == "test_label"
