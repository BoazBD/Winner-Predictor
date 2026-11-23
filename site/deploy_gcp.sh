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

# CPU and memory configuration
CPU_COUNT="1"  # smallest size; the app only reads a few hundred Firestore docs
MEMORY_SIZE="512Mi"

echo -e "${BLUE}🚀 Starting deployment to Google Cloud Run${NC}"
echo -e "${BLUE}🏗️  Project: ${GCP_PROJECT_ID}${NC}"
echo -e "${BLUE}🌍 Region: ${GCP_REGION}${NC}"
echo -e "${BLUE}💻 CPU: ${CPU_COUNT}${NC}"
echo -e "${BLUE}🧠 Memory: ${MEMORY_SIZE}${NC}"

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

# TZ: the app compares Winner kickoff times (Israel local) with datetime.now()
echo -e "${GREEN}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_TAG}" \
  --region="${GCP_REGION}" \
  --project="${GCP_PROJECT_ID}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --set-env-vars="FLASK_ENV=production,TZ=Asia/Jerusalem" \
  --allow-unauthenticated \
  --memory="${MEMORY_SIZE}" \
  --cpu="${CPU_COUNT}" \
  --min-instances="0" \
  --max-instances="1" \
  --cpu-throttling \
  --cpu-boost \
  --concurrency="80" \
  --timeout="120" \
  --revision-suffix="firestore-$(date +%Y%m%d-%H%M%S)" \
  --quiet

# Keep only the newest image in Artifact Registry (storage is billed per GB)
gcloud artifacts docker images list "us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}" \
  --include-tags --format="value(version,tags)" --project="${GCP_PROJECT_ID}" 2>/dev/null | awk '$2 !~ /latest/ {print $1}' | while read -r d; do
  gcloud artifacts docker images delete "us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}@${d}" --delete-tags --quiet --project="${GCP_PROJECT_ID}" >/dev/null 2>&1 || true
done

# Get the service URL
echo -e "${GREEN}✅ Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project="${GCP_PROJECT_ID}" --region="${GCP_REGION}" --format='value(status.url)')

echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}🏥 Health Check: ${SERVICE_URL}/health${NC}"

# Optional: Open the URL in browser (uncomment if desired)
# echo -e "${BLUE}🌐 Opening service in browser...${NC}"
# open "${SERVICE_URL}"

echo -e "${GREEN}🎉 Deployment successful!${NC}" 
# NOTE: the service was once "stopped" by switching it to manual scaling with 0 instances
# (it then answers 503 "Service is disabled"). To stop it again:   set annotation
#   run.googleapis.com/scalingMode: manual  +  run.googleapis.com/manualInstanceCount: '0'
# and to re-enable: scalingMode: automatic (gcloud run services describe --format=export | edit | gcloud run services replace).
