import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepoNotFoundError


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")
    return token


def push_to_hub(
    model_path: Path,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str = "Upload model",
) -> None:
    token = get_hf_token()
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepoNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_id, exist_ok=True)

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def create_model_card(
    repo_id: str,
    metrics: dict[str, float],
    model_name: str,
    dataset: str,
    architecture: str,
) -> str:
    metrics_lines = "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])

    return f"""---
license: apache-2.
tags:
- pytorch
- transformers
- {model_name}
- image-classification
- generated_from_trainer
datasets:
- {dataset}
---

# {model_name}

Fine-tuned {architecture} model for {dataset} classification.

## Metrics

{metrics_lines}

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```
"""
