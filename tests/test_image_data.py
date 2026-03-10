from pathlib import Path

from src.image_classifier.data import get_label_mapping, get_class_folders


def test_get_class_folders():
    data_dir = Path("data/image/sample_images")
    if data_dir.exists():
        folders = get_class_folders(data_dir)
        assert isinstance(folders, list)


def test_get_label_mapping():
    data_dir = Path("data/image/sample_images")
    if data_dir.exists() and any(get_class_folders(data_dir)):
        mapping = get_label_mapping(data_dir)
        assert isinstance(mapping, dict)
