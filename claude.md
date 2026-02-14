# TensorRT BenchLab — Claude Code Instructions

You are building a production-quality weekend project that benchmarks inference performance across:
1) PyTorch CPU
2) PyTorch CUDA GPU
3) TensorRT (via ONNX->TensorRT OR Torch-TensorRT)

## Non-negotiables (Definition of Done)
- Reproducible benchmark methodology:
  - warmup runs excluded
  - N measured iterations (configurable)
  - batch sizes: [1, 4, 8, 16] default
  - report p50, p95, throughput (req/s), stddev
  - fixed random seed and fixed input set
- Same preprocessing and postprocessing for all engines.
- Results stored as JSON with a stable schema.
- FastAPI backend exposes:
  - POST /bench/run (start benchmark run, returns run_id)
  - GET /bench/runs/{run_id} (results + metadata)
  - GET /telemetry/live (recent NVML samples)
- Frontend dashboard:
  - Run benchmark button + select model/mode/batch sizes
  - Charts for latency (p50/p95) vs batch size
  - Throughput chart
  - GPU util/mem chart over time
- Observability:
  - Prometheus metrics endpoint OR OpenTelemetry traces (prefer both if feasible)
- DevEx:
  - docker-compose up runs everything
  - makefile or scripts for common tasks
  - CI for lint + tests

## Engineering quality bar
- Clean architecture: engines abstracted behind an interface
- Typed Pydantic schemas for all API contracts
- Unit tests for methodology math and result schema validation
- No “magic numbers”: config in a single place
- All benchmark runs include environment metadata:
  - GPU name, driver version, CUDA version
  - torch + tensorrt versions
  - CPU model (if accessible)
  - timestamp + git commit hash

## Guardrails
- Prefer minimal dependencies.
- Avoid training. Use pretrained model weights.
- Keep runtime stable: avoid UI blocking; benchmark runs should be async/background task.

## Agent workflow
Work in PR-style chunks:
- Summarize what you changed
- List files edited/added
- Provide commands to run/verify
- Do not implement outside your assigned area if using subagents.

## Target environment assumptions
- NVIDIA GPU present with CUDA drivers
- Python 3.11+
- Node 20+
- Docker available
