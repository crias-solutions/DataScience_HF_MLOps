from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = ["text", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    return df


def get_label_mapping(labels: list[str]) -> dict[str, int]:
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def encode_labels(labels: list[str], label_mapping: dict[str, int]) -> list[int]:
    return [label_mapping[label] for label in labels]


def tokenize_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> dict[str, list[Any]]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def prepare_dataset(
    csv_path: str | Path,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    test_size: float = 0.2,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    df = load_csv_data(csv_path)

    labels = df["label"].tolist()
    texts = df["text"].tolist()

    label_mapping = get_label_mapping(labels)
    encoded_labels = encode_labels(labels, label_mapping)

    from sklearn.model_selection import train_test_split

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        random_state=42,
        stratify=encoded_labels,
    )

    train_encodings = tokenize_texts(train_texts, tokenizer, max_length)
    val_encodings = tokenize_texts(val_texts, tokenizer, max_length)

    return {
        "train_texts": train_texts,
        "val_texts": val_texts,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "train_encodings": train_encodings,
        "val_encodings": val_encodings,
        "label_mapping": label_mapping,
    }


def get_num_labels(label_mapping: dict[str, int]) -> int:
    return len(label_mapping)
