import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface BenchmarkRequest {
  model_name: string
  engine_type: 'pytorch_cpu' | 'pytorch_gpu' | 'tensorrt'
  batch_sizes: number[]
  num_iterations: number
  warmup_iterations: number
}

export interface BatchStats {
  batch_size: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_mean_ms: number
  latency_stddev_ms: number
  throughput_req_per_sec: number
}

export interface EnvironmentMetadata {
  gpu_name: string | null
  gpu_driver_version: string | null
  cuda_version: string | null
  torch_version: string
  tensorrt_version: string | null
  cpu_model: string | null
  timestamp: string
  git_commit: string | null
}

export interface BenchmarkResult {
  run_id: string
  model_name: string
  engine_type: string
  environment: EnvironmentMetadata
  results: BatchStats[]
  status: string
  error_message: string | null
  created_at: string
  completed_at: string | null
}

export interface TelemetrySample {
  timestamp: string
  gpu_utilization_percent: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_celsius: number | null
  power_usage_watts: number | null
}

export interface TelemetryResponse {
  samples: TelemetrySample[]
  device_name: string
}

export const startBenchmark = async (request: BenchmarkRequest) => {
  const response = await api.post('/bench/run', request)
  return response.data
}

export const getBenchmarkResult = async (runId: string): Promise<BenchmarkResult> => {
  const response = await api.get(`/bench/runs/${runId}`)
  return response.data
}

export const getTelemetry = async (): Promise<TelemetryResponse> => {
  const response = await api.get('/telemetry/live')
  return response.data
}
