# AWS Lambda (SAM) Deployment

This recipe shows how to deploy the Hugging Face MLOps API to AWS Lambda using AWS SAM.

## Prerequisites

- AWS CLI installed and configured
- SAM CLI installed (`brew install aws-sam-cli` or `pip install aws-sam-cli`)
- Docker installed (for building Lambda container images)

## Build and Deploy

### Option 1: Using Makefile

```bash
# Build the Docker image
make docker-build

# Package and deploy (interactive)
sam deploy --guided
```

### Option 2: Manual Deployment

1. Build the container image:
```bash
docker build -t hf-mlops-api:latest .
```

2. Tag for ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag hf-mlops-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/hf-mlops-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/hf-mlops-api:latest
```

3. Deploy SAM template:
```bash
sam deploy \
  --template-file template.yaml \
  --stack-name hf-mlops-api \
  --parameter-overrides Environment=prod HFToken=$HF_TOKEN \
  --capabilities CAPABILITY_IAM
```

## Testing

```bash
# Get API endpoint
API_URL=$(aws cloudformation describe-stacks --stack-name hf-mlops-api --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)

# Test text classification
curl -X POST "${API_URL}text/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Test image classification
curl -X POST "${API_URL}image/predict" \
  -F "file=@test_image.jpg"
```

## Notes

- AWS Lambda has a 10GB disk limit for container images
- Memory is limited to 10GB, which may affect large model inference
- Consider using Lambda@Edge or API Gateway for caching
