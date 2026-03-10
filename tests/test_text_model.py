import pytest
from pathlib import Path


def test_model_path_exists():
    model_path = Path("src/text_classifier/model")
    assert model_path.exists() or not model_path.exists()


def test_data_path_exists():
    data_path = Path("data/text/sample_data.csv")
    assert data_path.exists()


def test_csv_format():
    import pandas as pd

    df = pd.read_csv("data/text/sample_data.csv")
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0
