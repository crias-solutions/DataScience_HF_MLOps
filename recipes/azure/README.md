# Azure Container Apps Deployment

This recipe shows how to deploy the Hugging Face MLOps API to Azure Container Apps.

## Prerequisites

- Azure CLI installed (`az`)
- Docker installed

## Build and Deploy

### 1. Build and Push to ACR

```bash
# Set subscription
az account set --subscription YOUR_SUBSCRIPTION

# Create resource group
az group create --name hf-mlops-rg --location eastus

# Create container registry
az acr create --resource-group hf-mlops-rg --name hfmlopsacr --sku Standard

# Login to ACR
az acr login --name hfmlopsacr

# Build and tag
docker build -t hf-mlops-api:latest .
docker tag hf-mlops-api:latest hfmlopsacr.azurecr.io/hf-mlops-api:latest

# Push to ACR
docker push hfmlopsacr.azurecr.io/hf-mlops-api:latest
```

### 2. Deploy to Container Apps

```bash
# Create log analytics workspace
az monitor log-analytics workspace create --resource-group hf-mlops-rg --workspace-name hf-mlops-logs

# Get log analytics ID
LOG_ANALYTICS_ID=$(az monitor log-analytics workspace show --resource-group hf-mlops-rg --workspace-name hf-mlops-logs --query id --output tsv)

# Create container app environment
az containerapp env create \
  --name hf-mlops-env \
  --resource-group hf-mlops-rg \
  --location eastus

# Deploy container app
az containerapp create \
  --name hf-mlops-api \
  --resource-group hf-mlops-rg \
  --environment hf-mlops-env \
  --image hfmlopsacr.azurecr.io/hf-mlops-api:latest \
  --cpu 2.0 \
  --memory 4Gi \
  --port 8080 \
  --set-env-vars HF_TOKEN=$HF_TOKEN \
  --ingress external \
  --target-port 8080

# Enable CORS (optional)
az containerapp update \
  --name hf-mlops-api \
  --resource-group hf-mlops-rg \
  --min-replicas 1 \
  --max-replicas 5
```

## Testing

```bash
# Get service URL
SERVICE_URL=$(az containerapp show --name hf-mlops-api --resource-group hf-mlops-rg --query "properties.provisioningState" --output tsv)
FULL_URL="https://$(az containerapp show --name hf-mlops-api --resource-group hf-mlops-rg --query "properties.configuration.ingress.fqdn" --output tsv)"

# Test
curl -X POST "${FULL_URL}/text/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

## Notes

- Azure Container Apps has a cold start issue; consider setting min replicas to 1
- Traffic can be split between revisions for A/B testing
