import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)

from src.text_classifier.data import prepare_dataset
from src.shared.utils import ensure_dir, save_metrics, set_seed


class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


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
    model_path: str = "src/text_classifier/model",
    data_path: str = "data/text/sample_data.csv",
    output_path: str = "outputs/text_metrics.json",
    max_length: int = 128,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, float]:
    set_seed(seed)

    model_path = Path(model_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    import json

    with open(model_path / "label_mapping.json") as f:
        label_mapping = json.load(f)

    reverse_mapping = {v: k for k, v in label_mapping.items()}

    print("Preparing dataset...")
    dataset = prepare_dataset(data_path, tokenizer, max_length, test_size=0.2)

    val_dataset = TextDataset(dataset["val_encodings"], dataset["val_labels"])

    from transformers import Trainer

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
    parser.add_argument("--model", type=str, default="src/text_classifier/model")
    parser.add_argument("--data", type=str, default="data/text/sample_data.csv")
    parser.add_argument("--output", type=str, default="outputs/text_metrics.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
