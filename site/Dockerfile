FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV PORT=8080
ENV HOST=0.0.0.0

# Expose the port
EXPOSE 8080

# Run the application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"] 