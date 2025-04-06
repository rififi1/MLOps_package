# Use Python as base image
FROM python:3.11-alpine

# Set working directory in container
WORKDIR /app

# Install build dependencies needed for scientific Python packages
RUN apk add --no-cache \
    g++ \
    musl-dev \
    linux-headers \
    python3-dev \
    openblas-dev

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directory for model persistence if it doesn't exist
#RUN mkdir -p /app/models

# Make sure the data directory exists
#RUN mkdir -p /app/data

# Expose the port the api app runs on
EXPOSE 8000

# Command to run your FastAPI app
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "package/main.py"]