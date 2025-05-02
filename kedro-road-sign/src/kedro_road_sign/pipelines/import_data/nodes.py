import os

def import_data(dataset_name: str, download_path: str) -> str:
    """
    Import data from Kaggle if not already downloaded.

    Args:
        dataset_name (str): The Kaggle dataset identifier (e.g., "user/dataset-name").
        download_path (str): The path where the dataset should be downloaded.

    Returns:
        str: The path to the dataset files.
    """
    import kagglehub

    # Check if the dataset is already downloaded
    if not os.path.exists(download_path):
        print(f"Dataset not found at {download_path}. Downloading...")
        path = kagglehub.dataset_download(dataset_name, path=download_path)
        print("Dataset downloaded to:", path)
    else:
        print(f"Dataset already exists at {download_path}. Skipping download.")
        path = download_path

    return path