# Dockerfile
FROM mcr.microsoft.com/devcontainers/python:0-3.10

# Install basic utilities and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy files
COPY . .
COPY pyproject.toml requirements.txt /app

# Install package in editable mode
RUN pip install --upgrade pip && pip install -e /app

# Default command
CMD ["python"]