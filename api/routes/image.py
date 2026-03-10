import tempfile
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.image_classifier.inference import load_classifier, ImageClassifier


router = APIRouter()


def get_image_classifier() -> ImageClassifier:
    try:
        return load_classifier("src/image_classifier/model")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")


class ImagePrediction(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


@router.post("/predict", response_model=ImagePrediction)
async def predict_image(
    file: UploadFile = File(...),
    classifier: Annotated[ImageClassifier, Depends(get_image_classifier)],
) -> ImagePrediction:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = classifier.predict(tmp_path)

        return ImagePrediction(
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
