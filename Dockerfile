# Dockerfile
FROM mcr.microsoft.com/devcontainers/python:0-3.10

# Install basic utilities and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    npm \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ../pyproject.toml ../README.md ./
COPY ../src/ ./src/
# Install package in editable mode
RUN pip install --upgrade pip && pip install -e .

# Default command
CMD ["python"]