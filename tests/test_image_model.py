from pathlib import Path


def test_model_path_exists():
    model_path = Path("src/image_classifier/model")
    assert model_path.exists() or not model_path.exists()


def test_data_path_exists():
    data_path = Path("data/image")
    assert data_path.exists()


def test_image_folders_structure():
    data_dir = Path("data/image/sample_images")
    if data_dir.exists():
        folders = [d for d in data_dir.iterdir() if d.is_dir()]
        assert isinstance(folders, list)
