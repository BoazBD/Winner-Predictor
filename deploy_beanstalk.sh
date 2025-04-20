#!/bin/bash
set -e

# Configuration
BEANSTALK_APP="winner-site"
BEANSTALK_ENV="winner-site"
AWS_REGION="il-central-1"
S3_BUCKET="winner-site-data"
DEPLOYMENT_ZIP="site-deployment.zip"
VERSION_LABEL="v-$(date +%Y%m%d%H%M%S)"

echo "Starting Elastic Beanstalk deployment process..."

# Navigate to the site directory 
cd site

# Install dependencies if needed (uncomment if you need to install dependencies before deployment)
# pip install -r requirements.txt

# Create a deployment package
echo "Creating deployment package..."
zip -r "../$DEPLOYMENT_ZIP" . -x "*.git*" "*.DS_Store*" "__pycache__/*" "*.pyc" "*.pyo" "*.pyd" ".Python" "env/*" "venv/*" "ENV/*" "env.bak/*" "venv.bak/*"

# Navigate back to the root directory
cd ..

# Upload the deployment package to S3
echo "Uploading deployment package to S3..."
aws s3 cp "$DEPLOYMENT_ZIP" "s3://$S3_BUCKET/$DEPLOYMENT_ZIP" --region $AWS_REGION

# Create a new application version
echo "Creating new application version..."
aws elasticbeanstalk create-application-version \
    --application-name $BEANSTALK_APP \
    --version-label "$VERSION_LABEL" \
    --source-bundle S3Bucket="$S3_BUCKET",S3Key="$DEPLOYMENT_ZIP" \
    --region $AWS_REGION

# Update the environment to use the new version
echo "Updating Elastic Beanstalk environment..."
aws elasticbeanstalk update-environment \
    --environment-name $BEANSTALK_ENV \
    --version-label "$VERSION_LABEL" \
    --region $AWS_REGION

echo "Deployment complete! The update is now in progress."
echo "You can check the status of the deployment in the Elastic Beanstalk console."

# Clean up deployment package
echo "Cleaning up..."
rm "$DEPLOYMENT_ZIP"

echo "Done!" 