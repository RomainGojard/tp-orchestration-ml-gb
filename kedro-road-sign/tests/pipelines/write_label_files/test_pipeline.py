import pytest
import tempfile
import os
from pathlib import Path
from src.kedro_road_sign.pipelines.write_label_files.nodes import empty_input_model_folder, copy_files


def test_empty_input_model_folder_creates_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder = Path(tmpdir) / "to_empty"
        test_folder.mkdir()

        # Ajouter un fichier pour simuler du contenu existant
        dummy_file = test_folder / "dummy.txt"
        dummy_file.write_text("Some content")
        assert dummy_file.exists()

        # Appel de la fonction
        result = empty_input_model_folder(str(test_folder))

        # Le dossier doit exister mais être vide
        assert result == "folder_emptied"
        assert test_folder.exists()
        assert list(test_folder.iterdir()) == []


def test_empty_input_model_folder_creates_if_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder = Path(tmpdir) / "does_not_exist"

        # Le dossier n'existe pas avant
        assert not test_folder.exists()

        result = empty_input_model_folder(str(test_folder))

        assert result == "folder_emptied"
        assert test_folder.exists()
        assert list(test_folder.iterdir()) == []


def test_copy_files_copies_only_non_empty_txt_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "src"
        destination = Path(tmpdir) / "dest"
        source.mkdir()
        destination.mkdir()

        # Fichier texte non vide (doit être copié)
        (source / "file1.txt").write_text("content")

        # Fichier texte vide (ne doit PAS être copié)
        (source / "file2.txt").write_text("")

        # Fichier avec une autre extension (ne doit pas être copié)
        (source / "file3.csv").write_text("some,data")

        copy_files(str(source), str(destination))

        copied_files = list(destination.glob("*"))
        copied_names = [f.name for f in copied_files]

        assert "file1.txt" in copied_names
        assert "file2.txt" not in copied_names
        assert "file3.csv" not in copied_names
        assert len(copied_names) == 1
