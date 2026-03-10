import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.text_classifier.data import get_num_labels, prepare_dataset
from src.shared.utils import set_seed, ensure_dir


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


def train(
    data_path: str | Path,
    model_name: str = "distilbert-base-uncased",
    output_dir: str | Path = "src/text_classifier/model",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
) -> None:
    set_seed(seed)

    data_path = Path(data_path)
    output_dir = ensure_dir(output_dir)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Preparing dataset...")
    dataset = prepare_dataset(data_path, tokenizer, max_length)

    print(f"Number of labels: {get_num_labels(dataset['label_mapping'])}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=get_num_labels(dataset["label_mapping"]),
    )

    train_dataset = TextDataset(dataset["train_encodings"], dataset["train_labels"])
    val_dataset = TextDataset(dataset["val_encodings"], dataset["val_labels"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=str(ensure_dir(output_dir / "logs")),
        logging_steps=10,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Training model...")
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    import json

    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(dataset["label_mapping"], f)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default="src/text_classifier/model")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )
