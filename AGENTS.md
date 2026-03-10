# AGENTS.md

> This file provides context to OpenCode and other AI coding assistants about your project.

---

## Project Overview

**Name:** DataScience_HF_MLOps

**Description:** End-to-end Hugging Face MLOps scaffold with text and image classification pipelines, CI/CD automation, and deployment to Hugging Face Hub and Spaces.

**Type:** Python Application (MLOps)

---

## Tech Stack

- **Language:** Python 3.12
- **Package Manager:** pip
- **ML Framework:** PyTorch, Transformers, Diffusers
- **Data Processing:** Pandas, Pillow
- **Testing:** pytest
- **Linting:** Ruff
- **Formatting:** Ruff / Black
- **CI/CD:** GitHub Actions
- **Deployment:** Hugging Face Hub, Hugging Face Spaces, FastAPI

---

## Project Structure

```
project-root/
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI: lint + tests on every push
│       ├── cd-text.yml        # CD: train + deploy text classifier
│       └── cd-image.yml       # CD: train + deploy image classifier
├── .devcontainer/              # Codespaces config
├── data/
│   ├── text/                   # Text classification dataset
│   │   └── sample_data.csv
│   └── image/                  # Image classification dataset
│       └── sample_images/
├── src/
│   ├── __init__.py
│   ├── config.py               # Shared configuration
│   ├── text_classifier/
│   │   ├── __init__.py
│   │   ├── data.py             # Text data loading
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation
│   │   └── inference.py        # Inference logic
│   ├── image_classifier/
│   │   ├── __init__.py
│   │   ├── data.py             # Image data loading
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation
│   │   └── inference.py        # Inference logic
│   └── shared/
│       ├── __init__.py
│       ├── utils.py            # Shared utilities
│       └── hub.py              # HF Hub push utilities
├── app/
│   ├── text_app.py             # Gradio text classifier
│   └── image_app.py            # Gradio image classifier
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── text.py             # Text classification endpoint
│   │   └── image.py            # Image classification endpoint
│   └── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_text_data.py
│   ├── test_text_model.py
│   ├── test_image_data.py
│   ├── test_image_model.py
│   └── test_inference.py
├── configs/
│   ├── text-config.yaml
│   └── image-config.yaml
├── Makefile
├── requirements.txt
├── pyproject.toml
├── AGENTS.md
├── README.md
└── LICENSE
```

---

## Pipeline Overview

### Text Classification Pipeline
- **Model:** DistilBERT (fine-tuned for sentiment/classification)
- **Dataset:** CSV with `text` and `label` columns
- **Workflow:** Load data → Tokenize → Train → Evaluate → Push to Hub → Deploy to Space

### Image Classification Pipeline
- **Model:** ResNet50 (fine-tuned for classification)
- **Dataset:** Folders organized by class (e.g., `class_a/`, `class_b/`)
- **Workflow:** Load data → Preprocess → Train → Evaluate → Push to Hub → Deploy to Space

---

## Coding Standards

### Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters
- Use docstrings for public functions and classes

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `user_name` |
| Functions | snake_case | `get_user()` |
| Classes | PascalCase | `UserManager` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | _prefix | `_internal_method()` |

### Imports

```python
# Standard library
import os
import sys

# Third-party
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Local
from src.text_classifier import train
```

---

## Testing

### Run Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

### Test Naming

- Files: `test_<module>.py`
- Functions: `test_<function>_<scenario>()`

---

## Common Tasks

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Text Classification Pipeline Locally

```bash
make train-text
make evaluate-text
```

### Run Image Classification Pipeline Locally

```bash
make train-image
make evaluate-image
```

### Run Linter

```bash
ruff check .
```

### Format Code

```bash
ruff format .
```

---

## CI/CD Workflows

### CI Pipeline (ci.yml)
Runs on every push and PR:
1. Setup Python environment
2. Install dependencies
3. Run Ruff linting
4. Run type checking
5. Run pytest unit tests
6. Run model sanity tests

### CD Pipelines (cd-text.yml, cd-image.yml)
Run on merge to main:
1. Load and preprocess dataset
2. Fine-tune model
3. Compute evaluation metrics
4. Generate model card
5. Push model to Hugging Face Hub
6. Deploy to Hugging Face Space

---

## AI Assistant Guidelines

### Do

- Write clean, readable code
- Include type hints
- Add docstrings to public functions
- Write unit tests for new features
- Follow existing patterns in the codebase
- Use pure Python scripts (not PyTorch Lightning)

### Don't

- Remove existing tests without explanation
- Change coding style mid-project
- Add dependencies without justification
- Leave commented-out code

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Hugging Face API token for model push/deploy | Yes (for CD) |
| `ANTHROPIC_API_KEY` | Anthropic API key | No |
| `OPENAI_API_KEY` | OpenAI API key | No |

---

## Hugging Face Setup

### Required Secrets (GitHub Repository)

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add new repository secret:
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token (get from huggingface.co/settings/tokens)

### Getting a Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name: `GitHub Actions`
4. Role: **Write** (required for model push)
5. Copy the token and add to GitHub secrets

---

## Deployment Targets

### 1. Hugging Face Hub
- Model cards with metrics
- Versioned model weights
- Public or private models

### 2. Hugging Face Spaces (Gradio)
- Interactive web demos
- Auto-deployment on model update

### 3. FastAPI (Self-hosted)
- REST API for production
- Docker container support

---

## Notes

- Both text and image pipelines are independent
- Custom datasets can replace sample data in `data/` folder
- CI runs on every push; CD runs only on main branch merges

---

## Next Steps (Phase 2)

### 1. Docker Containerization

Add Dockerfile for self-hosted deployment:

- Create `Dockerfile` for FastAPI app
- Create `.dockerignore`
- Add docker commands to Makefile:
  - `make docker-build`
  - `make docker-run`
  - `make docker-push`

### 2. Enhanced Makefile Commands

Reference: [Python-MLOps-Cookbook Makefile](https://github.com/noahgift/Python-MLOps-Cookbook/blob/main/Makefile)

Add commands for:
- Model export/registry
- Local API testing
- Cloud deployment scripts

### 3. Cloud Deployment Options

Add deployment configs for:

- **AWS Lambda** (via SAM) - Serverless deployment
- **AWS CloudRun** - Containerized deployment
- **GCP CloudRun** - Containerized deployment
- **Azure Container Apps** - Containerized deployment

### 4. GitHub Actions Enhancements

- Add container build and push workflow
- Add AWS/GCP/Azure deployment workflows

### 5. Additional MLOps Features

- **MLflow integration** - Experiment tracking
- **Model monitoring** - Data drift detection
- **A/B testing** - Deploy multiple model versions

---

## Reference Repositories

For Phase 2 enhancements, refer to:

- [noahgift/Python-MLOps-Cookbook](https://github.com/noahgift/Python-MLOps-Cookbook) - Docker, Flask, multi-cloud deployment examples
