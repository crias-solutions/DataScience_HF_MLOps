import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.image_classifier.data import prepare_image_dataset
from src.shared.utils import set_seed, ensure_dir


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


def train(
    data_dir: str,
    model_name: str = "microsoft/resnet-50",
    output_dir: str = "src/image_classifier/model",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    image_size: int = 224,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    set_seed(seed)

    data_dir = Path(data_dir)
    output_dir = ensure_dir(output_dir)

    print(f"Loading model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(list(data_dir.iterdir())),
    )
    processor = AutoImageProcessor.from_pretrained(model_name)

    print("Preparing dataset...")
    dataset = prepare_image_dataset(data_dir, model_name, image_size, test_size, seed)

    train_dataset = ImageDataset(
        dataset["train_paths"],
        dataset["train_labels"],
        processor,
        image_size,
    )
    val_dataset = ImageDataset(
        dataset["val_paths"],
        dataset["val_labels"],
        processor,
        image_size,
    )

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

    print("Saving model and processor...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    import json

    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(dataset["label_mapping"], f)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="Path to image data directory"
    )
    parser.add_argument("--model", type=str, default="microsoft/resnet-50")
    parser.add_argument("--output-dir", type=str, default="src/image_classifier/model")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        test_size=args.test_size,
        seed=args.seed,
    )
