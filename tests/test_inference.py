import pytest
from pathlib import Path


def test_inference_module_exists():
    from src.text_classifier import inference

    assert hasattr(inference, "TextClassifier")
    assert hasattr(inference, "load_classifier")


def test_image_inference_module_exists():
    from src.image_classifier import inference

    assert hasattr(inference, "ImageClassifier")
    assert hasattr(inference, "load_classifier")
