#!/bin/bash

# Stop the script if any command fails
set -e

# AWS ECR login
echo "Logging into AWS ECR..."
aws ecr get-login-password --region il-central-1 | docker login --username AWS --password-stdin 105144480348.dkr.ecr.il-central-1.amazonaws.com

# Build Docker image
echo "Building Docker image..."
docker buildx build -t winner-api --platform linux/amd64 -f ./DockerfileApi .

# Tag the Docker image
echo "Tagging Docker image..."
docker tag winner-api:latest 105144480348.dkr.ecr.il-central-1.amazonaws.com/winner-api:latest

# Push the Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push 105144480348.dkr.ecr.il-central-1.amazonaws.com/winner-api:latest

echo "Docker image pushed successfully to ECR."

aws lambda update-function-code \
           --function-name arn:aws:lambda:il-central-1:105144480348:function:Winner-api-lambda \
           --image-uri 105144480348.dkr.ecr.il-central-1.amazonaws.com/winner-api:latest \
           --region il-central-1 > /dev/null

echo "Lambda image updated successfully."
