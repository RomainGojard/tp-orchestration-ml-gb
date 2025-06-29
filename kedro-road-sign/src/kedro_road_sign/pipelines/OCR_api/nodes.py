import pytesseract
import cv2
from typing import List, Dict
from difflib import SequenceMatcher
from pathlib import Path
import yaml, json

def prepare_ocr_data(images_path: str, labels_path: str, data_config: Dict, images_path_preprocessed: str) -> List:
    """Extrait les ROI 'panneaux' des images selon les prédictions du modèle YOLO."""
    files = Path(images_path).glob("*.png")
    rois = []

    if not data_config or 'names' not in data_config:
        raise ValueError(f"Le fichier YAML {data_config} est vide ou ne contient pas de 'names'.")
    labels_db = data_config['names']

    for file in files:
        label_path = labels_path + "/" + f"{file.stem}.txt"
        # Si le fichier de labels n'existe pas, on prend un label générique
        if not Path(label_path).exists():
            line = "1 0.5 0.5 1 1"
        else:
            with open(label_path, 'r') as f:
                line = f.read()
        labels = line.split(" ")

        image = cv2.imread(str(file))
        if image is None:
            raise FileNotFoundError(f"Image file {file} not found or could not be read.")

        x, y, width, height = map(float, labels[1:])  # coordonnées relatives

        label_id = labels[0]
        label = labels_db[int(label_id)] if int(label_id) < len(labels_db) else "unknown"
        if label == "unknown":
            print(f"Label {label_id} not found in labels_db, using 'unknown'.")

        img_h, img_w = image.shape[:2]
        abs_x = int((x - width / 2) * img_w)
        abs_y = int((y - height / 2) * img_h)
        abs_w = int(width * img_w)
        abs_h = int(height * img_h)

        x1 = max(0, abs_x)
        y1 = max(0, abs_y)
        x2 = min(img_w, abs_x + abs_w)
        y2 = min(img_h, abs_y + abs_h)
        
        cropped_roi = image[y1:y2, x1:x2]
        retouche = (opening(grayscale(remove_noise(cropped_roi))))
        
        preprocessed_image_path = Path(images_path_preprocessed) / file.name
        preprocessed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preprocessed_image_path), retouche)

        rois.append({
            "image": Path(preprocessed_image_path).as_posix(),
            "label": label,
            "roi": (x1, y1, x2, y2)
        })

    return rois

def configure_tesseract(path_cmd: str) -> None:
    pytesseract.pytesseract.tesseract_cmd = path_cmd

def ocr(rois: Dict, lang: str) -> Dict:
    """Exécute l'OCR sur les ROI"""

    for roi in rois:
        image = cv2.imread(roi['image'])
        text = pytesseract.image_to_string(image, lang=lang).strip()
        # sauvegarder les résultats dans un nouveau fichier texte que l'on créera dans le dossier 07_model_output/ocr_results
        if not text:
            text = "No text found"
        # Créer le dossier s'il n'existe pas
        Path("data/07_model_output/ocr_results").mkdir(parents=True, exist_ok=True)
        # Enregistrer le texte dans un fichier
        text_file_path = f"data/07_model_output/ocr_results/{Path(roi['image']).stem}.txt"
        with open(text_file_path, 'w') as f:
            f.write(f"Image: {roi['image']}, Predicted: {text}, Ground Truth: {roi['label']}, width: {roi['roi'][2]}, height: {roi['roi'][3]}, x: {roi['roi'][0]}, y: {roi['roi'][1]}")

        print(f"Image: {roi['image']}, Predicted: {text}, Ground Truth: {roi['label']}, width: {roi['roi'][2]}, height: {roi['roi'][3]}, x: {roi['roi'][0]}, y: {roi['roi'][1]}")
    
    return ()

def compute_cer(ref: str, hyp: str) -> int:
    """Calcule le nombre d'erreurs pour CER (Character Error Rate)."""
    matcher = SequenceMatcher(None, ref, hyp)
    return int(sum([max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != 'equal']))


### OCR Functions Utilities ###

import cv2
import numpy as np

# grayscale
def grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
	return cv2.medianBlur(image, 5)

# thresholding
def thresholding(image):
	return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
	kernel = np.ones((5, 5), np.uint8)
	return cv2.dilate(image, kernel, iterations=1)

# erosion
def erode(image):
	kernel = np.ones((5, 5), np.uint8)
	return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation
def opening(image):
	kernel = np.ones((5, 5), np.uint8)
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
	return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
	coords = np.column_stack(np.where(image > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return rotated

# template matching
def match_template(image, template):
	return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Rouge "faible" (proche de 0° sur le cercle HSV)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])

# Rouge "fort" (proche de 180° sur le cercle HSV)
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

