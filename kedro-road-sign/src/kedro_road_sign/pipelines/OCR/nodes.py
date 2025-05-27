import pytesseract
import cv2
from typing import List, Dict
from difflib import SequenceMatcher
from pathlib import Path
import yaml, json
from src.ocr_functions_utilities import *

def prepare_ocr_data(images_path: str, labels_path: str, data_config: Dict, images_path_preprocessed: str) -> List:
    """Extrait les ROI 'panneaux' des images selon les prédictions du modèle YOLO."""
    files = Path(images_path).glob("*.png")
    rois = json.loads("[]")  # initialiser une liste vide pour les ROIs

    # récupérer le tableau de labels dans le yaml au path data_config_path

    if not data_config or 'names' not in data_config:
        raise ValueError(f"Le fichier YAML {data_config} est vide ou ne contient pas de 'names'.")
    labels_db = data_config['names']

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

        # ✅ Convertir coordonnées relatives → absolues
        img_h, img_w = image.shape[:2]
        abs_x = int((x - width / 2) * img_w)
        abs_y = int((y - height / 2) * img_h)
        abs_w = int(width * img_w)
        abs_h = int(height * img_h)

        # ✅ Découpe de l'image à la ROI (avec clip pour éviter débordement)
        x1 = max(0, abs_x)
        y1 = max(0, abs_y)
        x2 = min(img_w, abs_x + abs_w)
        y2 = min(img_h, abs_y + abs_h)
        
        cropped_roi = image[y1:y2, x1:x2]  # Découpe de l'image
                
        """Traitement d'image"""
        retouche = (opening(grayscale(remove_noise(cropped_roi))))
        """Fin traitement"""
        
        # Enregistrer l'image traitée dans le dossier prétraité
        preprocessed_image_path = Path(images_path_preprocessed) / file.name
        preprocessed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preprocessed_image_path), retouche)
        
        #cv2.imshow("Image traitée", retouche)  # image peut être gray, thresh, etc.
        #cv2.waitKey(0)  # Attend une touche
        #cv2.destroyAllWindows()

        rois.append({
            "image": Path(preprocessed_image_path).as_posix(),  # chemin de l'image
            "label": label,
            "roi": (x1, y1, x2, y2) # plus utile
        })

    return rois

def configure_tesseract(path_cmd: str) -> None:
    pytesseract.pytesseract.tesseract_cmd = path_cmd

def evaluate_ocr(rois: Dict, lang: str) -> Dict:
    """Exécute l'OCR sur les ROI et évalue le CER."""
    predictions = []
    total_chars = 0
    total_errors = 0

    for roi in rois:
        image = cv2.imread(roi['image'])
        text = pytesseract.image_to_string(image, lang=lang).strip()
        print(f"Image: {roi['image']}, Predicted: {text}, Ground Truth: {roi['label']}, width: {roi['roi'][2]}, height: {roi['roi'][3]}, x: {roi['roi'][0]}, y: {roi['roi'][1]}")
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
