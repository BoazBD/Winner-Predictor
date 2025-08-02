#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define variables (replace with your actual values)
GCP_PROJECT_ID="winner-predictor-site"
GCP_REGION="us-central1"
SERVICE_NAME="winner-site"
ARTIFACT_REGISTRY_REPO="winner-site-repo"
IMAGE_TAG="us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}:latest"
SERVICE_ACCOUNT="winner-site-run-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Data source configuration (change this to switch between data sources)
DATA_SOURCE="firestore"  # Options: "firestore", "dynamodb", "sqlite"

echo -e "${BLUE}🚀 Starting deployment to Google Cloud Run${NC}"
echo -e "${BLUE}📊 Data Source: ${DATA_SOURCE}${NC}"
echo -e "${BLUE}🏗️  Project: ${GCP_PROJECT_ID}${NC}"
echo -e "${BLUE}🌍 Region: ${GCP_REGION}${NC}"

# Check if gcloud is authenticated
echo -e "${YELLOW}🔐 Checking Google Cloud authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with Google Cloud. Please run: gcloud auth login${NC}"
    exit 1
fi

# Set the project
echo -e "${YELLOW}📋 Setting project to ${GCP_PROJECT_ID}...${NC}"
gcloud config set project "${GCP_PROJECT_ID}"

# Configure Docker for Artifact Registry
echo -e "${YELLOW}🐳 Configuring Docker for Artifact Registry...${NC}"
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Build the Docker image for AMD64 architecture (required for Cloud Run)
echo -e "${GREEN}🏗️  Building Docker image for linux/amd64...${NC}"
docker build --platform linux/amd64 -t "${IMAGE_TAG}" .

# Push the image to Google Artifact Registry
echo -e "${GREEN}📤 Pushing image to Artifact Registry...${NC}"
docker push "${IMAGE_TAG}"

# Prepare environment variables based on data source
ENV_VARS="FLASK_ENV=production,DATA_SOURCE=${DATA_SOURCE}"

if [ "$DATA_SOURCE" = "firestore" ]; then
    echo -e "${BLUE}🔥 Configuring for Firestore...${NC}"
    ENV_VARS="${ENV_VARS},FIRESTORE_PROJECT_ID=${GCP_PROJECT_ID}"
    ENV_VARS="${ENV_VARS},FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION=profitable_predictions"
    ENV_VARS="${ENV_VARS},FIRESTORE_ALL_PREDICTIONS_COLLECTION=all_predictions"
    
    # Deploy to Cloud Run with Firestore configuration
    echo -e "${GREEN}🚀 Deploying to Cloud Run with Firestore...${NC}"
    gcloud run deploy "${SERVICE_NAME}" \
      --image="${IMAGE_TAG}" \
      --region="${GCP_REGION}" \
      --project="${GCP_PROJECT_ID}" \
      --service-account="${SERVICE_ACCOUNT}" \
      --set-env-vars="${ENV_VARS}" \
      --allow-unauthenticated \
      --memory="2Gi" \
      --cpu="1" \
      --max-instances="2" \
      --timeout="300" \
      --revision-suffix="firestore-$(date +%Y%m%d-%H%M%S)" \
      --quiet

elif [ "$DATA_SOURCE" = "dynamodb" ]; then
    echo -e "${BLUE}⚡ Configuring for DynamoDB...${NC}"
    ENV_VARS="${ENV_VARS},AWS_REGION=il-central-1"
    
    # Deploy to Cloud Run with DynamoDB configuration
    echo -e "${GREEN}🚀 Deploying to Cloud Run with DynamoDB...${NC}"
    gcloud run deploy "${SERVICE_NAME}" \
      --image="${IMAGE_TAG}" \
      --region="${GCP_REGION}" \
      --project="${GCP_PROJECT_ID}" \
      --service-account="${SERVICE_ACCOUNT}" \
      --set-env-vars="${ENV_VARS}" \
      --update-secrets="AWS_ACCESS_KEY_ID=aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest" \
      --allow-unauthenticated \
      --memory="1Gi" \
      --cpu="1" \
      --max-instances="10" \
      --timeout="300" \
      --revision-suffix="dynamodb-$(date +%Y%m%d-%H%M%S)" \
      --quiet

else
    echo -e "${BLUE}💾 Configuring for SQLite (local database)...${NC}"
    
    # Deploy to Cloud Run with SQLite configuration
    echo -e "${GREEN}🚀 Deploying to Cloud Run with SQLite...${NC}"
    gcloud run deploy "${SERVICE_NAME}" \
      --image="${IMAGE_TAG}" \
      --region="${GCP_REGION}" \
      --project="${GCP_PROJECT_ID}" \
      --service-account="${SERVICE_ACCOUNT}" \
      --set-env-vars="${ENV_VARS}" \
      --allow-unauthenticated \
      --memory="1Gi" \
      --cpu="1" \
      --max-instances="10" \
      --timeout="300" \
      --revision-suffix="sqlite-$(date +%Y%m%d-%H%M%S)" \
      --quiet
fi

# Get the service URL
echo -e "${GREEN}✅ Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project="${GCP_PROJECT_ID}" --region="${GCP_REGION}" --format='value(status.url)')

echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}🏥 Health Check: ${SERVICE_URL}/health${NC}"
echo -e "${GREEN}📊 Data Source: ${DATA_SOURCE}${NC}"

# Optional: Open the URL in browser (uncomment if desired)
# echo -e "${BLUE}🌐 Opening service in browser...${NC}"
# open "${SERVICE_URL}"

echo -e "${GREEN}🎉 Deployment successful!${NC}" 