import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getRun, cancelRun, RunRecord, EngineResult } from '../api/client'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import '../styles/RunDetailPage.css'

export default function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<RunRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchRun = async () => {
    if (!runId) return

    try {
      const data = await getRun(runId)
      setRun(data)
      setError(null)

      // Stop polling if run is complete
      if (['succeeded', 'failed', 'cancelled'].includes(data.status)) {
        setLoading(false)
      }
    } catch (err) {
      setError('Failed to fetch run: ' + (err as Error).message)
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRun()

    // Poll every 2 seconds if run is in progress
    const interval = setInterval(() => {
      if (run && ['queued', 'running'].includes(run.status)) {
        fetchRun()
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [runId])

  const handleCancelRun = async () => {
    if (!runId || !confirm('Cancel this run?')) return

    try {
      await cancelRun(runId)
      await fetchRun()
    } catch (err) {
      setError('Failed to cancel run: ' + (err as Error).message)
    }
  }

  if (!run) {
    return (
      <div className="run-detail-page">
        <div className="loading">Loading run details...</div>
      </div>
    )
  }

  // Get unique engine names from results
  const engineNames = Array.from(new Set(run.results.map(r => r.engine_name)))

  // Group results by batch size for latency chart
  const latencyChartData = run.batch_sizes.map((batchSize) => {
    const data: any = { batch_size: batchSize }
    run.results
      .filter((r) => r.batch_size === batchSize)
      .forEach((result) => {
        data[`${result.engine_name}_p50`] = result.latency_p50_ms
        data[`${result.engine_name}_p95`] = result.latency_p95_ms
      })
    return data
  })

  // Group results by batch size for throughput chart
  const throughputChartData = run.batch_sizes.map((batchSize) => {
    const data: any = { batch_size: batchSize }
    run.results
      .filter((r) => r.batch_size === batchSize)
      .forEach((result) => {
        data[result.engine_name] = result.throughput_req_per_sec
      })
    return data
  })

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'succeeded':
        return 'badge badge-success'
      case 'failed':
        return 'badge badge-error'
      case 'running':
        return 'badge badge-running'
      case 'queued':
        return 'badge badge-queued'
      case 'cancelled':
        return 'badge badge-cancelled'
      default:
        return 'badge'
    }
  }

  return (
    <div className="run-detail-page">
      <div className="page-header">
        <div>
          <button onClick={() => navigate('/')} className="btn btn-sm btn-secondary">
            ← Back
          </button>
          <h2>Run: {run.model_name}</h2>
          <p className="run-id">ID: {run.run_id}</p>
        </div>
        <div className="header-actions">
          <span className={getStatusBadgeClass(run.status)}>{run.status}</span>
          {['queued', 'running'].includes(run.status) && (
            <button onClick={handleCancelRun} className="btn btn-sm btn-danger">
              Cancel Run
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="run-info-grid">
        <div className="info-card">
          <h3>Configuration</h3>
          <table>
            <tbody>
              <tr>
                <td>Model:</td>
                <td>{run.model_name}</td>
              </tr>
              <tr>
                <td>Engines:</td>
                <td>{run.engines.join(', ')}</td>
              </tr>
              <tr>
                <td>Batch Sizes:</td>
                <td>{run.batch_sizes.join(', ')}</td>
              </tr>
              <tr>
                <td>Iterations:</td>
                <td>{run.num_iterations}</td>
              </tr>
              <tr>
                <td>Warmup:</td>
                <td>{run.warmup_iterations}</td>
              </tr>
              <tr>
                <td>Runner URL:</td>
                <td>{run.runner_url}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="info-card">
          <h3>Timing</h3>
          {run.timing ? (
            <table>
              <tbody>
                <tr>
                  <td>Total:</td>
                  <td>{run.timing.total_duration_sec.toFixed(2)}s</td>
                </tr>
                <tr>
                  <td>Model Load:</td>
                  <td>{run.timing.model_load_sec.toFixed(2)}s</td>
                </tr>
                <tr>
                  <td>Warmup:</td>
                  <td>{run.timing.warmup_duration_sec.toFixed(2)}s</td>
                </tr>
                <tr>
                  <td>Measurement:</td>
                  <td>{run.timing.measurement_duration_sec.toFixed(2)}s</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p>No timing data available</p>
          )}
        </div>

        {run.environment && (
          <div className="info-card">
            <h3>Environment</h3>
            <table>
              <tbody>
                {run.environment.gpu_name && (
                  <tr>
                    <td>GPU:</td>
                    <td>{run.environment.gpu_name}</td>
                  </tr>
                )}
                {run.environment.cuda_version && (
                  <tr>
                    <td>CUDA:</td>
                    <td>{run.environment.cuda_version}</td>
                  </tr>
                )}
                {run.environment.tensorrt_version && (
                  <tr>
                    <td>TensorRT:</td>
                    <td>{run.environment.tensorrt_version}</td>
                  </tr>
                )}
                {run.environment.sanity_check_passed !== null && (
                  <tr>
                    <td>Sanity Check:</td>
                    <td>{run.environment.sanity_check_passed ? '✅ Passed' : '❌ Failed'}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {run.status === 'succeeded' && run.results.length > 0 && (
        <>
          <div className="chart-section">
            <h3>Latency (p50 & p95) vs Batch Size</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={latencyChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="batch_size" label={{ value: 'Batch Size', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {engineNames.map((engine, idx) => (
                  <Line
                    key={`${engine}_p50`}
                    type="monotone"
                    dataKey={`${engine}_p50`}
                    stroke={['#8884d8', '#82ca9d', '#ffc658'][idx % 3]}
                    name={`${engine} p50`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-section">
            <h3>Throughput vs Batch Size</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={throughputChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="batch_size" label={{ value: 'Batch Size', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Throughput (req/s)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {engineNames.map((engine, idx) => (
                  <Bar
                    key={engine}
                    dataKey={engine}
                    fill={['#8884d8', '#82ca9d', '#ffc658'][idx % 3]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="results-table-section">
            <h3>Detailed Results</h3>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Batch</th>
                  <th>p50 (ms)</th>
                  <th>p95 (ms)</th>
                  <th>Mean (ms)</th>
                  <th>Stddev (ms)</th>
                  <th>Throughput (req/s)</th>
                </tr>
              </thead>
              <tbody>
                {run.results.map((result, idx) => (
                  <tr key={idx}>
                    <td>{result.engine_name}</td>
                    <td>{result.batch_size}</td>
                    <td>{result.latency_p50_ms.toFixed(2)}</td>
                    <td>{result.latency_p95_ms.toFixed(2)}</td>
                    <td>{result.latency_mean_ms.toFixed(2)}</td>
                    <td>{result.latency_stddev_ms.toFixed(2)}</td>
                    <td>{result.throughput_req_per_sec.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {run.status === 'failed' && run.error_message && (
        <div className="error-section">
          <h3>Error Details</h3>
          <pre>{run.error_message}</pre>
          {run.error_stack && (
            <details>
              <summary>Stack Trace</summary>
              <pre>{run.error_stack}</pre>
            </details>
          )}
        </div>
      )}

      {run.telemetry && run.telemetry.samples.length > 0 && (
        <div className="telemetry-section">
          <h3>GPU Telemetry</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={run.telemetry.samples}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="t_ms"
                label={{ value: 'Time (ms)', position: 'insideBottom', offset: -5 }}
                tickFormatter={(val) => (val / 1000).toFixed(1) + 's'}
              />
              <YAxis label={{ value: 'GPU Util (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                labelFormatter={(val) => `Time: ${(val / 1000).toFixed(1)}s`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="gpu_utilization_percent"
                stroke="#8884d8"
                name="GPU Utilization %"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
