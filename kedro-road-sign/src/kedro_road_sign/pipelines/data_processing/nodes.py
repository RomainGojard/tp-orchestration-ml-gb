import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a  file.

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


def preprocess_data(data: pd.DataFrame, output_dir: str) -> None:
    """
    Preprocess data and convert it to YOLO format.

    Args:
        data (pd.DataFrame): Input data as a pandas DataFrame.
        output_dir (str): Directory to save the YOLO-formatted data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Example conversion to YOLO format
    for index, row in data.iterrows():
        yolo_format = f"{row['class']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}\n"
        output_file = os.path.join(output_dir, f"{row['image_id']}.txt")
        with open(output_file, "w") as f:
            f.write(yolo_format)
    print(f"Data preprocessed and saved to {output_dir}")

