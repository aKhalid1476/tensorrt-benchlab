import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getRun, cancelRun, RunRecord } from '../api/client'
import {
  AreaChart,
  Area,
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

const CHART_COLORS = ['#f97316', '#6366f1', '#10b981', '#ec4899']

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#0d1117',
  border: '1px solid rgba(148, 163, 184, 0.15)',
  borderRadius: '10px',
  color: '#e2e8f0',
  fontSize: '13px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
}

const CHART_LEGEND_STYLE = { paddingTop: '12px', color: '#94a3b8', fontSize: '13px' }
const AXIS_TICK = { fill: '#64748b', fontSize: 12 }
const AXIS_LABEL_STYLE = { fill: '#475569', fontSize: 12 }

export default function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<RunRecord | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchRun = async () => {
    if (!runId) return

    try {
      const data = await getRun(runId)
      setRun(data)
      setError(null)

    } catch (err) {
      setError('Failed to fetch run: ' + (err as Error).message)
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
            <p className="chart-subtitle">Lower is better — p95 shows tail latency</p>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={latencyChartData} margin={{ top: 10, right: 24, left: 16, bottom: 36 }}>
                <defs>
                  {engineNames.map((engine, idx) => (
                    <linearGradient key={engine} id={`grad-${idx}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS[idx % CHART_COLORS.length]} stopOpacity={0.18} />
                      <stop offset="95%" stopColor={CHART_COLORS[idx % CHART_COLORS.length]} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="0" stroke="rgba(148,163,184,0.08)" vertical={false} />
                <XAxis
                  dataKey="batch_size"
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={{ stroke: 'rgba(148,163,184,0.15)' }}
                  label={{ value: 'Batch Size', position: 'insideBottom', offset: -20, style: AXIS_LABEL_STYLE }}
                />
                <YAxis
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', offset: 8, style: AXIS_LABEL_STYLE }}
                />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} cursor={{ stroke: 'rgba(148,163,184,0.15)', strokeWidth: 1 }} />
                <Legend wrapperStyle={CHART_LEGEND_STYLE} />
                {engineNames.map((engine, idx) => (
                  <Area
                    key={`${engine}_p50`}
                    type="monotone"
                    dataKey={`${engine}_p50`}
                    stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                    strokeWidth={2.5}
                    fill={`url(#grad-${idx})`}
                    dot={{ r: 4, fill: CHART_COLORS[idx % CHART_COLORS.length], strokeWidth: 0 }}
                    activeDot={{ r: 6, strokeWidth: 0 }}
                    name={`${engine} p50`}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-section">
            <h3>Throughput vs Batch Size</h3>
            <p className="chart-subtitle">Higher is better — images processed per second</p>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={throughputChartData} margin={{ top: 10, right: 24, left: 16, bottom: 36 }} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="0" stroke="rgba(148,163,184,0.08)" vertical={false} />
                <XAxis
                  dataKey="batch_size"
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={{ stroke: 'rgba(148,163,184,0.15)' }}
                  label={{ value: 'Batch Size', position: 'insideBottom', offset: -20, style: AXIS_LABEL_STYLE }}
                />
                <YAxis
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: 'req / sec', angle: -90, position: 'insideLeft', offset: 8, style: AXIS_LABEL_STYLE }}
                />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} cursor={{ fill: 'rgba(148,163,184,0.05)' }} />
                <Legend wrapperStyle={CHART_LEGEND_STYLE} />
                {engineNames.map((engine, idx) => (
                  <Bar
                    key={engine}
                    dataKey={engine}
                    fill={CHART_COLORS[idx % CHART_COLORS.length]}
                    radius={[6, 6, 0, 0]}
                    maxBarSize={64}
                    name={engine}
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
          <p className="chart-subtitle">Utilization sampled at 200ms intervals during the run</p>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={run.telemetry.samples} margin={{ top: 10, right: 24, left: 16, bottom: 36 }}>
              <defs>
                <linearGradient id="grad-gpu" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="0" stroke="rgba(148,163,184,0.08)" vertical={false} />
              <XAxis
                dataKey="t_ms"
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={{ stroke: 'rgba(148,163,184,0.15)' }}
                tickFormatter={(val) => (val / 1000).toFixed(1) + 's'}
                label={{ value: 'Time (s)', position: 'insideBottom', offset: -20, style: AXIS_LABEL_STYLE }}
              />
              <YAxis
                domain={[0, 100]}
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={false}
                label={{ value: 'GPU Util %', angle: -90, position: 'insideLeft', offset: 8, style: AXIS_LABEL_STYLE }}
              />
              <Tooltip
                contentStyle={CHART_TOOLTIP_STYLE}
                cursor={{ stroke: 'rgba(148,163,184,0.15)', strokeWidth: 1 }}
                labelFormatter={(val) => `t = ${(val / 1000).toFixed(1)}s`}
              />
              <Area
                type="monotone"
                dataKey="gpu_utilization_percent"
                stroke="#6366f1"
                strokeWidth={2}
                fill="url(#grad-gpu)"
                name="GPU Util %"
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
