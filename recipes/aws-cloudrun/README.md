# AWS CloudRun Deployment

This recipe shows how to deploy the Hugging Face MLOps API to AWS CloudRun.

## Prerequisites

- AWS CLI installed and configured
- Docker installed

## Build and Deploy

### 1. Build and Push to ECR

```bash
# Get AWS account ID and region
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Create ECR repository
aws ecr create-repository --repository-name hf-mlops-api || true

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build and tag
docker build -t hf-mlops-api:latest .
docker tag hf-mlops-api:latest ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/hf-mlops-api:latest

# Push to ECR
docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/hf-mlops-api:latest
```

### 2. Deploy to CloudRun

```bash
aws cloudrun deploy hf-mlops-api \
  --image ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/hf-mlops-api:latest \
  --platform managed \
  --region $AWS_REGION \
  --cpu 2 \
  --memory 4Gi \
  --port 8080 \
  --set-env-vars HF_TOKEN=$HF_TOKEN \
  --allow-unauthenticated
```

## Testing

```bash
# Get service URL
SERVICE_URL=$(aws cloudrun describe-service --name hf-mlops-api --region $AWS_REGION --query "status.url" --output text)

# Test
curl -X POST "${SERVICE_URL}/text/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```
