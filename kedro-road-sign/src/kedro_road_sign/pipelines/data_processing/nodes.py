import pandas as pd
import os
from pathlib import Path
import shutil

def convert_to_yolo_format(row, img_w, img_h):
    x_center = (row['Roi.X1'] + row['Roi.X2']) / 2 / img_w
    y_center = (row['Roi.Y1'] + row['Roi.Y2']) / 2 / img_h
    width = (row['Roi.X2'] - row['Roi.X1']) / img_w
    height = (row['Roi.Y2'] - row['Roi.Y1']) / img_h
    return f"{int(row['ClassId'])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_labels(csv_path: Path, image_base_path: Path, output_dir: Path, subset: str) -> None:
    df = pd.read_csv(csv_path)
    images_dir = output_dir / 'images' / subset
    labels_dir = output_dir / 'labels' / subset
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        img_path = image_base_path / row['Path']
        label_path = labels_dir / f"{Path(row['Path']).stem}.txt"

        # Convert annotations
        label_line = convert_to_yolo_format(row, row['Width'], row['Height'])
        with open(label_path, 'a') as f:
            f.write(label_line + "\n")

        # Copy image
        dest_image_path = images_dir / Path(row['Path']).name
        if not dest_image_path.exists():
            shutil.copy(img_path, dest_image_path)
