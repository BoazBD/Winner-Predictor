#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables (replace if your project/region/repo differs)
GCP_PROJECT_ID="winner-predictor-site"
GCP_REGION="us-central1"
SERVICE_NAME="winner-site"
IMAGE_TAG="us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/winner-site-repo/${SERVICE_NAME}:latest"
SERVICE_ACCOUNT="winner-site-run-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com" # The dedicated SA we created

# Build the Docker image for AMD64 architecture
echo "Building Docker image..."
docker build --platform linux/amd64 -t "${IMAGE_TAG}" .

# Push the image to Google Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push "${IMAGE_TAG}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_TAG}" \
  --region="${GCP_REGION}" \
  --project="${GCP_PROJECT_ID}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --set-env-vars="AWS_REGION=il-central-1,USE_DYNAMODB=1,FLASK_ENV=production" \
  --update-secrets="AWS_ACCESS_KEY_ID=aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest" \
  --revision-suffix="deploy-$(date +%Y%m%d-%H%M%S)" \ # Force new revision
  --quiet # Suppress interactive prompts

echo "Deployment complete. Service URL:"
gcloud run services describe "${SERVICE_NAME}" --project="${GCP_PROJECT_ID}" --region="${GCP_REGION}" --format='value(status.url)' 