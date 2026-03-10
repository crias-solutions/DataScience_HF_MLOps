from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor


def get_class_folders(data_dir: Path) -> list[Path]:
    return [d for d in data_dir.iterdir() if d.is_dir()]


def get_label_mapping(data_dir: Path) -> dict[str, int]:
    folders = get_class_folders(data_dir)
    folders_sorted = sorted(folders, key=lambda x: x.name)
    return {folder.name: idx for idx, folder in enumerate(folders_sorted)}


def load_image_paths(
    data_dir: Path, label_mapping: dict[str, int]
) -> tuple[list[Path], list[int]]:
    image_paths = []
    labels = []

    for folder in get_class_folders(data_dir):
        label = label_mapping[folder.name]
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                image_paths.append(img_file)
                labels.append(label)

    return image_paths, labels


def split_dataset(
    image_paths: list[Path],
    labels: list[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        image_paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )


def prepare_image_dataset(
    data_dir: Path,
    model_name: str = "microsoft/resnet-50",
    image_size: int = 224,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    data_dir = Path(data_dir)

    label_mapping = get_label_mapping(data_dir)
    image_paths, labels = load_image_paths(data_dir, label_mapping)

    train_paths, val_paths, train_labels, val_labels = split_dataset(
        image_paths, labels, test_size, seed
    )

    processor = AutoImageProcessor.from_pretrained(model_name)

    return {
        "train_paths": train_paths,
        "val_paths": val_paths,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "processor": processor,
        "image_size": image_size,
        "label_mapping": label_mapping,
    }


def preprocess_image(image_path: Path, processor, image_size: int = 224) -> Any:
    image = Image.open(image_path).convert("RGB")
    return processor(images=image, return_tensors="pt")
