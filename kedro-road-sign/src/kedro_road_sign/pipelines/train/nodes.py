# src/my_project/pipelines/yolo_pipeline/nodes.py

from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def train_yolov8(datapath: str) -> dict:
    """Entraîne un modèle YOLOv8."""
    model = YOLO("yolov8n.pt")  # ou 'yolov8s.pt', etc.

    # ✅ Extraire le chemin YAML depuis le dict
    data_yaml_path = datapath

    results = model.train(
        data=data_yaml_path,  # ✅ ICI : string, pas dict
        epochs=10,
        imgsz=640,
        project="outputs/yolo_train",  # évite le "outputs/rain" typo 😉
        name="yolov8_exp1",
        exist_ok=True,
    )
    logger.info("Training completed.")

    return {
        "epoch": results.epoch,
        "best_fitness": results.best_fitness or 0.0,
    }