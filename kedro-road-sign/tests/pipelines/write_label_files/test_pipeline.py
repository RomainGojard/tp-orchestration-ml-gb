"""
This is a boilerplate test file for pipeline 'write_label_files'
generated using Kedro 0.19.13.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import unittest
import tempfile
from pathlib import Path
import os
from src.kedro_road_sign.pipelines.write_label_files.nodes import empty_input_model_folder, copy_files

def test_empty_input_model_folder_creates_empty_folder(self):
  with tempfile.TemporaryDirectory() as tmpdir:
    test_dir = Path(tmpdir) / "to_empty"
    test_dir.mkdir()

    # Ajouter des fichiers pour simuler du contenu
    (test_dir / "dummy.txt").write_text("test")

    # Appel de la fonction
    result = empty_input_model_folder(str(test_dir))

    # Le dossier doit exister mais être vide
    self.assertEqual(result, "folder_emptied")
    self.assertTrue(test_dir.exists())
    self.assertEqual(len(list(test_dir.iterdir())), 0)

def test_copy_files_copies_non_empty_txt(self):
  with tempfile.TemporaryDirectory() as tmpdir:
    src = Path(tmpdir) / "src"
    dst = Path(tmpdir) / "dst"
    src.mkdir()
    dst.mkdir()

    # Fichier non vide (doit être copié)
    (src / "file1.txt").write_text("contenu")
    # Fichier vide (ne doit PAS être copié)
    (src / "file2.txt").write_text("")

    # Appel de la fonction
    copy_files(str(src), str(dst))

    copied_files = list(dst.glob("*.txt"))
    copied_filenames = [f.name for f in copied_files]

    self.assertIn("file1.txt", copied_filenames)
    self.assertNotIn("file2.txt", copied_filenames)
    self.assertEqual(len(copied_filenames), 1)
