from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.text_classifier.inference import load_classifier, TextClassifier


router = APIRouter()


def get_text_classifier() -> TextClassifier:
    try:
        return load_classifier("src/text_classifier/model")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")


class TextInput(BaseModel):
    text: str


class TextPrediction(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


@router.post("/predict", response_model=TextPrediction)
def predict_text(
    input_data: TextInput,
    classifier: Annotated[TextClassifier, Depends(get_text_classifier)],
) -> TextPrediction:
    result = classifier.predict(input_data.text)

    return TextPrediction(
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )


@router.post("/predict_batch")
def predict_text_batch(
    texts: list[str],
    classifier: Annotated[TextClassifier, Depends(get_text_classifier)],
) -> list[TextPrediction]:
    results = classifier.predict_batch(texts)

    return [
        TextPrediction(
            predicted_label=r["predicted_label"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
        )
        for r in results
    ]
