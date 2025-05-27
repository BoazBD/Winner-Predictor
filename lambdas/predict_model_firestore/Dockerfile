FROM public.ecr.aws/lambda/python:3.9

# Copy requirements.txt first to leverage Docker caching for dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install the specified packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other necessary files from the build context
# This includes lambda_function.py, team_translations.py, service-account.json, etc.
COPY . ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ] 