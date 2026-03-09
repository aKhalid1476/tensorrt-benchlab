# TensorRT BenchLab Runner

GPU-based benchmark execution service.

## Setup

```bash
# Install dependencies
pip install -e .

# Install contracts
cd ../contracts && pip install -e . && cd ../runner
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Endpoints

- `GET /health` - Health check
- `GET /version` - Version and environment info
- `POST /execute` - Execute benchmark (synchronous)
