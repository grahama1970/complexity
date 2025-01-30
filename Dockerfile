# Base image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

# Set up environment
ENV PYTHONPATH=/app/server/src:$PYTHONPATH

# Add version label and build arguments
ARG DOCKERHUB_USER=yourusername
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0.0"
LABEL description="Complexity training container for RunPod"

# Add health check and metadata
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Remove ENTRYPOINT to allow override in compose file
# (We'll handle this in the command instead)