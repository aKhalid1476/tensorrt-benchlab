import { useState, useEffect } from 'react'
import BenchmarkControls from '../components/BenchmarkControls'
import ResultsChart from '../components/ResultsChart'
import TelemetryChart from '../components/TelemetryChart'
import { BenchmarkResult, getBenchmarkResult } from '../api/client'
import './BenchmarkDashboard.css'

function BenchmarkDashboard() {
  const [currentRun, setCurrentRun] = useState<BenchmarkResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleBenchmarkStart = (runId: string) => {
    setIsLoading(true)
    pollBenchmarkResult(runId)
  }

  const pollBenchmarkResult = async (runId: string) => {
    const poll = async () => {
      try {
        const result = await getBenchmarkResult(runId)
        setCurrentRun(result)

        if (result.status === 'completed' || result.status === 'failed') {
          setIsLoading(false)
        } else {
          setTimeout(poll, 2000) // Poll every 2 seconds
        }
      } catch (error) {
        console.error('Failed to fetch benchmark result:', error)
        setIsLoading(false)
      }
    }

    await poll()
  }

  return (
    <div className="dashboard">
      <section className="dashboard-section">
        <h2>Benchmark Controls</h2>
        <BenchmarkControls onBenchmarkStart={handleBenchmarkStart} isLoading={isLoading} />
      </section>

      {currentRun && (
        <>
          <section className="dashboard-section">
            <h2>Results: {currentRun.model_name} ({currentRun.engine_type})</h2>
            {currentRun.status === 'running' && (
              <div className="status-indicator running">
                <span className="spinner"></span>
                Benchmark running...
              </div>
            )}
            {currentRun.status === 'completed' && (
              <div className="status-indicator completed">✓ Completed</div>
            )}
            {currentRun.status === 'failed' && (
              <div className="status-indicator failed">
                ✗ Failed: {currentRun.error_message}
              </div>
            )}
            {currentRun.results.length > 0 && <ResultsChart results={currentRun.results} />}
          </section>

          <section className="dashboard-section">
            <h2>Environment</h2>
            <div className="env-info">
              <div className="env-item">
                <strong>GPU:</strong> {currentRun.environment.gpu_name || 'N/A'}
              </div>
              <div className="env-item">
                <strong>CUDA:</strong> {currentRun.environment.cuda_version || 'N/A'}
              </div>
              <div className="env-item">
                <strong>PyTorch:</strong> {currentRun.environment.torch_version}
              </div>
              <div className="env-item">
                <strong>CPU:</strong> {currentRun.environment.cpu_model || 'N/A'}
              </div>
            </div>
          </section>
        </>
      )}

      <section className="dashboard-section">
        <h2>GPU Telemetry</h2>
        <TelemetryChart />
      </section>
    </div>
  )
}

export default BenchmarkDashboard
