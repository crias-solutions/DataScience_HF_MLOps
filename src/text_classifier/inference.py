import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextClassifier:
    def __init__(self, model_path: str = "src/text_classifier/model"):
        model_path = Path(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        with open(model_path / "label_mapping.json") as f:
            self.label_mapping = json.load(f)

        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

    def predict(self, text: str) -> dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )

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

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.predict(text) for text in texts]


def load_classifier(model_path: str = "src/text_classifier/model") -> TextClassifier:
    return TextClassifier(model_path)
