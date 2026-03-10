# DataScience_HF_MLOps

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-F77F00?style=for-the-badge&logo=mozilla)](https://opensource.org/licenses/MPL-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=flat-square)](https://huggingface.co/)

End-to-end Hugging Face MLOps scaffold with text and image classification pipelines, CI/CD automation, and deployment to Hugging Face Hub and Spaces.

---

## What is this?

A production-ready MLOps scaffold for building, training, and deploying machine learning models using Hugging Face tools. This project provides:

- **Text Classification Pipeline** - Fine-tune DistilBERT for text classification
- **Image Classification Pipeline** - Fine-tune ResNet for image classification
- **Full CI/CD** - Automated testing and deployment via GitHub Actions
- **Multiple Deployment Options** - Hugging Face Hub, Spaces, and FastAPI

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    End-to-End MLOps Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Text Class   │    │ Image Class  │    │ Shared Core  │        │
│  │ Pipeline     │    │ Pipeline     │    │ (config,     │        │
│  │              │    │              │    │  utils)      │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│           │                  │                   │                  │
│           ▼                  ▼                   ▼                  │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              GitHub Actions CI/CD                          │      │
│  │  • CI: Ruff lint → Pytest → Model sanity tests            │      │
│  │  CD: Train → Evaluate → Push to HF Hub → Deploy Spaces    │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Deployment Targets:                                                │
│    1. Hugging Face Hub (model card + weights)                       │
│    2. Hugging Face Spaces (Gradio demo)                            │
│    3. FastAPI (self-hosted)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
DataScience_HF_MLOps/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + Tests on every push
│       ├── cd-text.yml        # Train + Deploy text classifier
│       └── cd-image.yml       # Train + Deploy image classifier
├── .devcontainer/
├── data/
│   ├── text/                  # Custom text dataset
│   │   └── sample_data.csv
│   └── image/                 # Custom image dataset
│       └── sample_images/
├── src/
│   ├── config.py              # Shared configuration
│   ├── text_classifier/      # Text classification pipeline
│   │   ├── data.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── image_classifier/     # Image classification pipeline
│   │   ├── data.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   └── shared/                # Shared utilities
│       ├── utils.py
│       └── hub.py
├── app/                       # Gradio demos
│   ├── text_app.py
│   └── image_app.py
├── api/                       # FastAPI application
│   ├── main.py
│   ├── routes/
│   │   ├── text.py
│   │   └── image.py
│   └── requirements.txt
├── tests/                     # Test suite
├── configs/                    # Configuration files
├── Makefile
├── requirements.txt
├── pyproject.toml
├── AGENTS.md
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Hugging Face account
- GitHub account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/crias-solutions/DataScience_HF_MLOps.git
cd DataScience_HF_MLOps
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Hugging Face Setup

1. Get your HF token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Set it as an environment variable:
```bash
export HF_TOKEN=your_token_here
```

---

## Usage

### Text Classification Pipeline

#### Train
```bash
python -m src.text_classifier.train --data data/text/sample_data.csv
```

#### Evaluate
```bash
python -m src.text_classifier.evaluate --model text-classifier
```

#### Run Gradio Demo
```bash
python -m app.text_app
```

### Image Classification Pipeline

#### Train
```bash
python -m src.image_classifier.train --data data/image
```

#### Evaluate
```bash
python -m src.image_classifier.evaluate --model image-classifier
```

#### Run Gradio Demo
```bash
python -m app.image_app
```

### Using Makefile

```bash
make install          # Install dependencies
make train-text       # Train text classifier
make train-image      # Train image classifier
make evaluate-text    # Evaluate text classifier
make evaluate-image  # Evaluate image classifier
make lint             # Run linter
make test             # Run tests
```

---

## CI/CD Workflows

### CI Pipeline (ci.yml)
Runs on every push and PR:
- Ruff linting
- Type checking
- Unit tests
- Model sanity tests

### CD Pipeline (cd-text.yml / cd-image.yml)
Runs on merge to main:
- Loads and preprocesses dataset
- Fine-tunes model
- Computes evaluation metrics
- Generates model card
- Pushes model to Hugging Face Hub
- Deploys to Hugging Face Space

---

## Deployment

### Hugging Face Hub

Models are automatically pushed to HF Hub with:
- Model weights
- Model card (metrics, usage)
- Evaluation results

### Hugging Face Spaces

Gradio demos deploy automatically to:
- `crias-solutions/text-classifier-space`
- `crias-solutions/image-classifier-space`

### FastAPI

Run locally:
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Customization

### Using Your Own Dataset

**Text Classification:**
Replace `data/text/sample_data.csv` with your own CSV:
```csv
text,label
"I love this!",positive
"This is bad",negative
```

**Image Classification:**
Organize images in folders by class:
```
data/image/
├── positive/
│   ├── image1.jpg
│   └── image2.jpg
└── negative/
    ├── image3.jpg
    └── image4.jpg
```

### Changing Models

Edit `src/text_classifier/train.py` or `src/image_classifier/train.py` to use different models:
- Text: `bert-base-uncased`, `roberta-base`, etc.
- Image: `vit-base-patch16-224`, `efficientnet-b0`, etc.

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
