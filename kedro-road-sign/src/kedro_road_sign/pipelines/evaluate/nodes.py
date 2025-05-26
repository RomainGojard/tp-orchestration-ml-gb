from ultralytics import YOLO

def evaluate_yolov8(model_path: str, data_yaml_path: str) -> dict:
    """
    Évalue un modèle YOLOv8 entraîné.

    Args:
        model_path: Chemin vers le modèle entraîné (.pt).
        data_yaml_path: Chemin vers le fichier YAML de dataset.

    Returns:
        Un dictionnaire contenant les métriques d'évaluation principales (mAP, précision, rappel...).
    """

    model = YOLO(model_path)

    # Évaluation du modèle sur le dataset de validation
    metrics = model.val(data=data_yaml_path)
    
    # Tu peux ajuster ce dictionnaire selon les métriques que tu veux sortir
    return {
        "metrics/mAP50": metrics.box.map50,
        "metrics/mAP50-95": metrics.box.map,
        "metrics/precision": metrics.box.precision,
        "metrics/recall": metrics.box.recall,
    }
