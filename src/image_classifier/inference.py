import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor


class ImageClassifier:
    def __init__(self, model_path: str = "src/image_classifier/model"):
        model_path = Path(model_path)

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval()

        with open(model_path / "label_mapping.json") as f:
            self.label_mapping = json.load(f)

        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

    def predict(self, image_path: str | Path) -> dict[str, float]:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        probs = probabilities.numpy()
        predictions = {
            self.reverse_mapping[i]: float(prob) for i, prob in enumerate(probs)
        }

        predicted_label = self.reverse_mapping[int(torch.argmax(probabilities))]

        return {
            "predicted_label": predicted_label,
            "probabilities": predictions,
            "confidence": float(torch.max(probabilities)),
        }

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict[str, float]]:
        return [self.predict(path) for path in image_paths]


def load_classifier(model_path: str = "src/image_classifier/model") -> ImageClassifier:
    return ImageClassifier(model_path)
