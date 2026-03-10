# DataScience_HF_MLOps Guide

A comprehensive guide to setting up and using this Hugging Face MLOps scaffold.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GitHub Secrets Setup](#github-secrets-setup)
3. [Local Development Setup](#local-development-setup)
4. [Training Pipelines](#training-pipelines)
5. [Running Demos](#running-demos)
6. [Deployment](#deployment)
7. [Validation Steps](#validation-steps)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.12+ | `python --version` |
| Git | Latest | `git --version` |
| Docker | Latest | `docker --version` |
| pip | Latest | `pip --version` |

### Install Python 3.12

```bash
# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3.12-dev

# Windows
# Download from python.org
```

---

## GitHub Secrets Setup

### Step 1: Get Your Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Configure:
   - **Name:** `GitHub Actions`
   - **Role:** **Write** (required for model push)
4. Copy the token

### Step 2: Add Secrets to GitHub

1. Go to your repository: `https://github.com/crias-solutions/DataScience_HF_MLOps`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Add the following secrets:

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `HF_TOKEN` | Hugging Face API token with Write access | Yes |
| `HF_REPO_ID_TEXT` | Model repo for text classifier (e.g., `your-username/text-classifier`) | Yes |
| `HF_SPACE_TEXT` | Space repo for text classifier Gradio app | Yes |
| `HF_REPO_ID_IMAGE` | Model repo for image classifier | Yes |
| `HF_SPACE_IMAGE` | Space repo for image classifier Gradio app | Yes |

### Validation

```bash
# Verify secrets are set (GitHub UI)
# Go to: Settings → Secrets and variables → Actions
# You should see all 5 secrets listed
```

---

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/crias-solutions/DataScience_HF_MLOps.git
cd DataScience_HF_MLOps
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Validation

```bash
# Check installed packages
python -c "import torch; import transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__)"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
```

---

## Training Pipelines

### Text Classification Pipeline

#### Step 1: Prepare Your Data

Edit `data/text/sample_data.csv` with your data:

```csv
text,label
"I love this product!",positive
"This is terrible",negative
```

#### Step 2: Train the Model

```bash
make train-text
```

Or with custom parameters:

```bash
python -m src.text_classifier.train \
  --data data/text/sample_data.csv \
  --model distilbert-base-uncased \
  --num-epochs 3 \
  --batch-size 16
```

#### Step 3: Evaluate

```bash
make evaluate-text
```

### Image Classification Pipeline

#### Step 1: Prepare Your Data

Organize images in folders:

```
data/image/
├── positive/
│   ├── image1.jpg
│   └── image2.png
└── negative/
    ├── image3.jpg
    └── image4.png
```

#### Step 2: Train the Model

```bash
make train-image
```

Or with custom parameters:

```bash
python -m src.image_classifier.train \
  --data data/image \
  --model microsoft/resnet-50 \
  --num-epochs 3
```

#### Step 3: Evaluate

```bash
make evaluate-image
```

### Validation

```bash
# Check model files were created
ls -la src/text_classifier/model/
ls -la src/image_classifier/model/

# Check metrics were generated
cat outputs/text_metrics.json
cat outputs/image_metrics.json
```

Expected: Directories should contain model files (`config.json`, `model.safetensors`, etc.)

---

## Running Demos

### Gradio Text Classifier App

```bash
make run-text-app
```

Then open: `http://localhost:7860`

### Gradio Image Classifier App

```bash
make run-image-app
```

Then open: `http://localhost:7861`

### FastAPI Server

```bash
make run-api
```

Then test endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Text prediction
curl -X POST http://localhost:8000/text/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

### Validation

```bash
# Test API health endpoint
curl -s http://localhost:8000/health | python -m json.tool

# Expected output:
# {"status": "healthy"}
```

---

## Deployment

### Option 1: Hugging Face Hub (Automatic via CI/CD)

The CD workflows automatically deploy when you push to `main`:

1. **Push changes to main:**
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. **Watch the workflow:**
   - Go to **Actions** tab
   - Select the running workflow
   - Monitor training and deployment progress

3. **Check deployed models:**
   - Text: `https://huggingface.co/{HF_REPO_ID_TEXT}`
   - Image: `https://huggingface.co/{HF_REPO_ID_IMAGE}`

### Option 2: Docker Deployment

#### Build Docker Image

```bash
make docker-build
```

#### Run Locally

```bash
# Set your HF token
export HF_TOKEN=your_token_here

make docker-run
```

Then test: `http://localhost:8080`

#### Push to Registry

```bash
make docker-push
```

### Option 3: Cloud Deployment

See deployment recipes:

| Cloud | Location |
|-------|----------|
| AWS Lambda | `recipes/aws-lambda-sam/` |
| AWS CloudRun | `recipes/aws-cloudrun/` |
| GCP CloudRun | `recipes/gcp-cloudrun/` |
| Azure | `recipes/azure/` |

### Validation

```bash
# Check GitHub Actions status
# Go to: https://github.com/crias-solutions/DataScience_HF_MLOps/actions

# Verify workflow runs completed successfully (green checkmarks)
```

---

## CI/CD Workflows

### CI Pipeline (ci.yml)

Runs on **every push and PR**:

1. **Lint** - Code quality checks
2. **Type Check** - Type validation
3. **Test** - Unit tests
4. **Model Sanity** - Verify model loading

### CD Pipeline (cd-text.yml / cd-image.yml)

Runs on **merge to main**:

1. Load and preprocess dataset
2. Fine-tune model
3. Compute evaluation metrics
4. Push model to Hugging Face Hub
5. Deploy to Hugging Face Space

### Docker Pipeline (docker.yml)

Runs on **Dockerfile changes**:

1. Build Docker image
2. Push to GitHub Container Registry

### Validation

```bash
# Check workflow status
gh run list

# Or via GitHub UI
# https://github.com/crias-solutions/DataScience_HF_MLOps/actions
```

---

## Validation Steps Summary

### Pre-Deployment Checklist

| Step | Command | Expected Result |
|------|---------|-----------------|
| Python version | `python --version` | Python 3.12.x |
| Dependencies | `pip list \| grep -E "torch\|transformers"` | Packages listed |
| Lint | `make lint` | No errors |
| Tests | `make test` | All tests pass |
| Train text | `make train-text` | Model saved |
| Train image | `make train-image` | Model saved |
| API | `curl http://localhost:8000/health` | `{"status": "healthy"}` |

### Post-Deployment Checklist

| Step | URL/Command | Expected Result |
|------|-------------|----------------|
| Text model on HF | `https://huggingface.co/{HF_REPO_ID_TEXT}` | Model page exists |
| Image model on HF | `https://huggingface.co/{HF_REPO_ID_IMAGE}` | Model page exists |
| Text Space | `https://huggingface.co/{HF_SPACE_TEXT}` | Gradio app works |
| Image Space | `https://huggingface.co/{HF_SPACE_IMAGE}` | Gradio app works |

---

## Troubleshooting

### Common Issues

#### 1. HF_TOKEN not set

**Error:** `HFToken is not set`

**Solution:**
```bash
export HF_TOKEN=your_huggingface_token
```

#### 2. Model not found

**Error:** `OSError: Model not found`

**Solution:** 
- Ensure you've trained the model first: `make train-text`
- Or pull from Hugging Face Hub in your code

#### 3. Docker build fails

**Error:** `failed to solve`

**Solution:**
```bash
# Clean Docker cache
docker builder prune

# Rebuild
docker build --no-cache -t hf-mlops-api:latest .
```

#### 4. GitHub Actions fails

**Error:** Workflow fails

**Solution:**
- Check **Actions** tab for error details
- Verify all secrets are set correctly
- Check Python version compatibility

### Get Help

- Check GitHub Issues
- Review workflow logs in **Actions** tab
- Test locally before pushing

---

## Quick Reference

### Makefile Commands

```bash
# Development
make install          # Install dependencies
make train-text       # Train text classifier
make train-image      # Train image classifier
make evaluate-text    # Evaluate text classifier
make evaluate-image  # Evaluate image classifier

# Testing
make test            # Run tests
make lint            # Run linter

# Demos
make run-text-app    # Text Gradio app
make run-image-app   # Image Gradio app
make run-api         # FastAPI server

# Docker
make docker-build    # Build container
make docker-run      # Run container
make docker-push     # Push to registry
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token |
| `PYTHONUNBUFFERED` | Enable unbuffered output |

---

## Next Steps

After getting started, consider:

1. **Customize datasets** - Replace sample data with your own
2. **Try different models** - Change `--model` parameter
3. **Add cloud deployment** - Use recipes in `recipes/`
4. **Explore Phase 3** - MLflow, monitoring, A/B testing (see AGENTS.md)

---

For more details, see [README.md](README.md) and [AGENTS.md](AGENTS.md).
