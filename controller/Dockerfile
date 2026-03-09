# TensorRT BenchLab Controller Dockerfile
# Mac M2 compatible (no CUDA dependencies)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy contracts package first (dependency)
COPY contracts /app/contracts
RUN cd /app/contracts && pip install --no-cache-dir -e .

# Copy controller code
COPY controller/pyproject.toml controller/README.md /app/controller/
COPY controller/app /app/controller/app

# Install controller dependencies
RUN cd /app/controller && pip install --no-cache-dir -e .

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BENCHLAB_LOG_LEVEL=INFO
ENV DATABASE_URL=sqlite:///app/data/benchlab.db

# Run the application
WORKDIR /app/controller
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
