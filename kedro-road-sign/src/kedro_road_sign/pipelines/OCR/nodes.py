import pytesseract
import cv2
from typing import List, Dict
from difflib import SequenceMatcher
from pathlib import Path
import yaml

def prepare_ocr_data(images_path: str, labels_path: str, data_config_path: str) -> List:
    """Extrait les ROI 'panneaux' des images selon les détections YOLO."""
    files = Path(images_path).glob("*.png")
    rois = []

    # récupérer le tableau de labels dans le yaml au path data_config_path

    with open(data_config_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    if not data_yaml or 'names' not in data_yaml:
        raise ValueError(f"Le fichier YAML {data_config_path} est vide ou ne contient pas de 'names'.")
    labels_db = data_yaml['names']

    for file in files:
        label_path = labels_path + "/" + f"{file.stem}.txt"
        # lire les roi dans le fichier de labels
        with open(label_path, 'r') as f:
            line = f.read()
        labels = line.split(" ")

        image = cv2.imread(str(file))
        if image is None:
            raise FileNotFoundError(f"Image file {file} not found or could not be read.")

        #print(f"Processing file: {line}, found {labels}")

        # le format est [class_id, x1, y1, x2, y2]
        x, y, width, height = map(float, labels[1:])  # prendre les coordonnées comme des float

        label_id = labels[0]
        label = labels_db[int(label_id)] if int(label_id) < len(labels_db) else "unknown"
        if label == "unknown":
            print(f"Label {label_id} not found in labels_db, using 'unknown'.")

        # convertir en valeurs absolues
        x = x * image.shape[1]  # largeur de l'image
        y = y * image.shape[0]  # hauteur de l'image
        width = width * image.shape[1]
        height = height * image.shape[0]

        

        rois.append({
            "image": image,
            "label": label,
            "roi": (x, y, width, height)
        })
        
    return rois

def configure_tesseract(path_cmd: str) -> None:
    pytesseract.pytesseract.tesseract_cmd = path_cmd

def evaluate_ocr(rois: List, lang: str) -> Dict:
    """Exécute l'OCR sur les ROI et évalue le CER."""
    predictions = []
    total_chars = 0
    total_errors = 0

    for roi in rois:
        text = pytesseract.image_to_string(roi['image'], lang=lang).strip()
        predictions.append(text)
        ground_truth =  roi['label'].strip()
        total_chars += len(ground_truth)
        total_errors += compute_cer(ground_truth, text)

    cer = total_errors / total_chars if total_chars > 0 else 1.0
    
    return {
        "ocr/cer": cer,
        "ocr/nb_samples": len(rois)
    }

def compute_cer(ref: str, hyp: str) -> int:
    """Calcule le nombre d'erreurs pour CER (Character Error Rate)."""
    matcher = SequenceMatcher(None, ref, hyp)
    return int(sum([max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != 'equal']))
