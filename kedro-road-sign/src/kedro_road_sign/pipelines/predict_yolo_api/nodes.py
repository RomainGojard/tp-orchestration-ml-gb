from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

def predict_yolov8(model_path: str, images_yolo_predict_input_path: str, data_config: dict, yolo_predict_output_dir: str) -> None:
    model = YOLO(model_path)
    os.makedirs(yolo_predict_output_dir, exist_ok=True)

    image_paths = list(Path(images_yolo_predict_input_path).glob("*.png"))  # Adjust the glob pattern if needed
    class_names = data_config.get('names', [])

    for image_path in image_paths:
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        pred = model(img)

        if pred[0].boxes is None or len(pred[0].boxes) == 0:
            print(f"No predictions for {image_path}.")
            continue
        else:
            print(f"Predictions for {image_path}: {pred}")
        
        txt_lines = []

        box = pred[0].boxes[0]  # Assuming we are only interested in the first prediction
        cls_id = int(box.cls[0])
        roi = box.xywh  # xywh format: [x_center, y_center, width, height]

        # Convert to YOLO format
        x_center = roi[0] / width
        y_center = roi[1] / height
        box_width = roi[2] / width
        box_height = roi[3] / height

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        txt_lines.append(line)

        # Save prediction in .txt with same name as image
        image_name = Path(image_path).stem
        txt_path = Path(yolo_predict_output_dir) / f"{image_name}.txt"
        #print(f"Prediction : {txt_lines} for {image_name}.txt")
        with open(txt_path, "w") as f:
            f.write(txt_lines[0] if txt_lines else "")  # Write the first line if exists, else empty