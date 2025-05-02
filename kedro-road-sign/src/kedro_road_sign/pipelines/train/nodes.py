import os
from ultralytics import YOLO

def train_yolo_model(data_dir: str, model_config: str, epochs: int, output_dir: str) -> str:
    """
    Train a YOLO model on the given dataset.

    Args:
        data_dir (str): Path to the directory containing YOLO-formatted data.
        model_config (str): Path to the YOLO model configuration file or pretrained model.
        epochs (int): Number of epochs to train the model.
        output_dir (str): Directory to save the trained model.

    Returns:
        str: Path to the saved YOLO model.
    """
    # Initialize YOLO model
    model = YOLO(model_config)

    # Train the model
    model.train(data=data_dir, epochs=epochs, project=output_dir, name="yolo_training")

    # Get the path to the best model weights
    best_model_path = os.path.join(output_dir, "yolo_training", "weights", "best.pt")
    print(f"Model trained and saved at: {best_model_path}")

    return best_model_path