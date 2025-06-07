FROM public.ecr.aws/lambda/python:3.10-x86_64

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code and service account
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY service-account.json ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ] 