import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.image_classifier.data import prepare_image_dataset
from src.shared.utils import ensure_dir, save_metrics, set_seed


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        processor,
        image_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(self.labels[idx])}


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


def evaluate(
    model_path: str = "src/image_classifier/model",
    data_dir: str = "data/image",
    output_path: str = "outputs/image_metrics.json",
    image_size: int = 224,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, float]:
    set_seed(seed)

    model_path = Path(model_path)
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    print(f"Loading model from {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    import json

    with open(model_path / "label_mapping.json") as f:
        label_mapping = json.load(f)

    print("Preparing dataset...")
    dataset = prepare_image_dataset(
        data_dir, str(model_path), image_size, test_size=0.2, seed=seed
    )

    val_dataset = ImageDataset(
        dataset["val_paths"],
        dataset["val_labels"],
        processor,
        image_size,
    )

    training_args = TrainingArguments(
        output_dir=str(ensure_dir(model_path / "eval")),
        per_device_eval_batch_size=batch_size,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Evaluating model...")
    metrics = trainer.evaluate()

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    save_metrics(metrics, output_path)
    print(f"\nMetrics saved to {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="src/image_classifier/model")
    parser.add_argument("--data", type=str, default="data/image")
    parser.add_argument("--output", type=str, default="outputs/image_metrics.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_dir=args.data,
        output_path=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
