"""
This is a boilerplate pipeline 'write_label_files'
generated using Kedro 0.19.13
"""
from pathlib import Path
import shutil
import os

def empty_input_model_folder(folder_path: str) -> str:

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    return "folder_emptied"

def copy_files(source_dir: str, dest_dir: str, _trigger: str = None) -> None:
  for file in Path(source_dir).glob("*.txt"):
    if file.is_file() and file.stat().st_size > 0:
      shutil.copy2(file, Path(dest_dir) / file.name)