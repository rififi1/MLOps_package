# Use Python as base image (alpine is the smallest image I found)
FROM python:3.11-alpine

# Set working directory in container
WORKDIR /app

# Install build dependencies needed for scientific Python packages (not necessary when using a slim python image)
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

# Expose the port on which the api server runs
EXPOSE 8000

# TODO: try without, not 100% it's actually useful
VOLUME ["/app/models"]

# Command to run the app
CMD ["python", "main.py"]
