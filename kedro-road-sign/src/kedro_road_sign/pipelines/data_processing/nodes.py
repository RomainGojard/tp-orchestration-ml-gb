import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}")
    return data


def preprocess_data(data: pd.DataFrame) -> dict:
    """
    Preprocess data and convert it to YOLO format.

    Args:
        data (pd.DataFrame): Input data as a pandas DataFrame.

    Returns:
        dict: A dictionary where keys are filenames and values are DataFrames.
    """
    yolo_data = {}

    # Convert data to YOLO format
    for index, row in data.iterrows():
        # Normalize bounding box coordinates
        x_center = (row["Roi.X1"] + row["Roi.X2"]) / (2 * row["Width"])
        y_center = (row["Roi.Y1"] + row["Roi.Y2"]) / (2 * row["Height"])
        width = (row["Roi.X2"] - row["Roi.X1"]) / row["Width"]
        height = (row["Roi.Y2"] - row["Roi.Y1"]) / row["Height"]

        yolo_format = pd.DataFrame(
            [[row['ClassId'], x_center, y_center, width, height]],
            columns=["ClassId", "x_center", "y_center", "width", "height"]
        )

        # Use the image path (without extension) as the key
        filename = os.path.splitext(row["Path"])[0]
        yolo_data[filename] = yolo_format

    return yolo_data

