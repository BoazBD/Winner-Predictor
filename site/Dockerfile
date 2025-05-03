FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy local code to the container image
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (Cloud Run default is 8080)
EXPOSE 8080

# Set environment variables (can be overridden by Cloud Run)
ENV FLASK_ENV=production
ENV USE_DYNAMODB=1 
# Add other necessary ENV vars here if they aren't set in Cloud Run

# Set the default command (Python app) to be executed by the entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"] 