# TensorRT BenchLab Runner Dockerfile
# NVIDIA CUDA base for TensorRT support

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy contracts package first (dependency)
COPY contracts /app/contracts
RUN cd /app/contracts && pip install --no-cache-dir -e .

# Copy runner code
COPY runner/pyproject.toml runner/README.md /app/runner/
COPY runner/app /app/runner/app

# Install runner dependencies
RUN cd /app/runner && pip install --no-cache-dir -e .

# Create cache directory
RUN mkdir -p /app/cache/tensorrt

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BENCHLAB_LOG_LEVEL=INFO

# Run the application
WORKDIR /app/runner
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
