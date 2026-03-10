# GCP CloudRun Deployment

This recipe shows how to deploy the Hugging Face MLOps API to Google Cloud CloudRun.

## Prerequisites

- gcloud CLI installed and configured
- Docker installed

## Build and Deploy

### 1. Build and Push to GCR

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Get project ID
PROJECT_ID=$(gcloud config get-value project)

# Build and tag
docker build -t hf-mlops-api:latest .
docker tag hf-mlops-api:latest gcr.io/${PROJECT_ID}/hf-mlops-api:latest

# Push to GCR
docker push gcr.io/${PROJECT_ID}/hf-mlops-api:latest
```

### 2. Deploy to CloudRun

```bash
gcloud run deploy hf-mlops-api \
  --image gcr.io/${PROJECT_ID}/hf-mlops-api:latest \
  --platform managed \
  --region us-central1 \
  --cpu 2 \
  --memory 4Gi \
  --set-env-vars HF_TOKEN=$HF_TOKEN \
  --allow-unauthenticated
```

## Testing

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe hf-mlops-api --platform managed --region us-central1 --format "value(status.url)")

# Test
curl -X POST "${SERVICE_URL}/text/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```
