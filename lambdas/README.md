# Winner Lambda Functions

This directory contains the Lambda functions used by the Winner project.

## Structure

- `get_results_from_winner/` - Lambda function to fetch game results from Winner
- `update_results/` - Lambda function to update game results
- `predict_model/` - Lambda function for prediction models

## Deployment

Each Lambda function has its own deployment script:

- `deploy_get_results.sh` - Deploy the get_results_from_winner Lambda
- `deploy_update_results.sh` - Deploy the update_results Lambda
- `deploy_predict_model.sh` - Deploy the predict_model Lambda
- `deploy_all.sh` - Deploy all Lambda functions at once

### Requirements

Some Lambda functions require environment variables to be set before deployment:

- `get_results_from_winner`: Requires the `PROXY_URL` environment variable to be set.

### Examples

To deploy a single Lambda function:

```bash
# Set required environment variables first
export PROXY_URL=your_proxy_url

# Deploy a specific Lambda
./deploy_get_results.sh
./deploy_update_results.sh
./deploy_predict_model.sh
```

To deploy all Lambda functions:

```bash
# Set required environment variables first
export PROXY_URL=your_proxy_url

# Deploy all Lambda functions
./deploy_all.sh
``` 