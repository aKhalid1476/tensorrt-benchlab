import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ==============================================================================
// Request/Response Types (matching contracts package)
// ==============================================================================

export type EngineType = 'pytorch_cpu' | 'pytorch_cuda' | 'tensorrt'
export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface RunCreateRequest {
  runner_url: string
  model_name: string
  engines: EngineType[]
  batch_sizes: number[]
  num_iterations: number
  warmup_iterations: number
  client_run_key?: string
}

export interface RunCreateResponse {
  run_id: string
  status: RunStatus
  message: string
}

export interface EngineResult {
  engine_name: string
  batch_size: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_mean_ms: number
  latency_stddev_ms: number
  throughput_req_per_sec: number
  timing_breakdown: TimingBreakdown | null
}

export interface EnvironmentMetadata {
  gpu_name: string | null
  gpu_driver_version: string | null
  cuda_version: string | null
  torch_version: string
  tensorrt_version: string | null
  python_version: string
  cpu_model: string | null
  timestamp: string
  git_commit: string | null
  sanity_check_passed: boolean | null
}

export interface TelemetrySample {
  t_ms: number  // Relative timestamp from run start
  gpu_utilization_percent: number | null
  memory_used_mb: number | null
  memory_total_mb: number | null
  temperature_celsius: number | null
  power_usage_watts: number | null
}

export interface TelemetryResponse {
  samples: TelemetrySample[]
  device_name: string
  run_id: string
}

export interface TimingBreakdown {
  total_duration_sec: number
  model_load_sec: number
  warmup_duration_sec: number
  measurement_duration_sec: number
}

export interface RunRecord {
  run_id: string
  status: RunStatus
  created_at: string
  started_at: string | null
  completed_at: string | null
  runner_url: string
  model_name: string
  engines: EngineType[]
  batch_sizes: number[]
  num_iterations: number
  warmup_iterations: number
  client_run_key: string | null
  environment: EnvironmentMetadata | null
  results: EngineResult[]
  telemetry: TelemetryResponse | null
  timing: TimingBreakdown | null
  error_message: string | null
  error_stack: string | null
}

export interface RunListResponse {
  runs: RunRecord[]
  total: number
}

// ==============================================================================
// API Functions
// ==============================================================================

export const createRun = async (request: RunCreateRequest): Promise<RunCreateResponse> => {
  const response = await api.post('/runs', request)
  return response.data
}

export const getRun = async (runId: string): Promise<RunRecord> => {
  const response = await api.get(`/runs/${runId}`)
  return response.data
}

export const listRuns = async (limit: number = 50): Promise<RunListResponse> => {
  const response = await api.get(`/runs?limit=${limit}`)
  return response.data
}

export const cancelRun = async (runId: string): Promise<{ run_id: string; status: string; message: string }> => {
  const response = await api.post(`/runs/${runId}/cancel`)
  return response.data
}

export const getRunReport = async (runId: string, save: boolean = false): Promise<{ markdown: string; file_path?: string }> => {
  const response = await api.get(`/runs/${runId}/report.md?save=${save}`)
  return response.data
}

export const healthCheck = async (): Promise<{ status: string; service: string }> => {
  const response = await api.get('/health')
  return response.data
}
