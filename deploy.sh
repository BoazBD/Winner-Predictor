#!/bin/bash

# Exit if any command fails
set -e

# AWS ECR login
echo "Logging into AWS ECR..."
aws ecr get-login-password --region il-central-1 | docker login --username AWS --password-stdin 105144480348.dkr.ecr.il-central-1.amazonaws.com

# Building the Docker image
echo "Building Docker image..."
docker buildx build --platform linux/amd64 -f ./Dockerfile -t winner_scraper .

# Tagging the Docker image
echo "Tagging Docker image..."
docker tag winner_scraper:latest 105144480348.dkr.ecr.il-central-1.amazonaws.com/winner_scraper:latest

# Pushing the Docker image to AWS ECR
echo "Pushing Docker image to AWS ECR..."
docker push 105144480348.dkr.ecr.il-central-1.amazonaws.com/winner_scraper:latest

echo "Deployment completed successfully."
