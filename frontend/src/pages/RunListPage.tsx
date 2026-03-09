import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listRuns, RunRecord } from '../api/client'
import '../styles/RunListPage.css'

export default function RunListPage() {
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchRuns = async () => {
    try {
      setLoading(true)
      const response = await listRuns(50)
      setRuns(response.runs)
      setError(null)
    } catch (err) {
      setError('Failed to fetch runs: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRuns()
    // Poll every 5 seconds for updates
    const interval = setInterval(fetchRuns, 5000)
    return () => clearInterval(interval)
  }, [])

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

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'N/A'
    const date = new Date(dateStr)
    return date.toLocaleString()
  }

  const formatDuration = (start: string | null, end: string | null) => {
    if (!start || !end) return 'N/A'
    const duration = new Date(end).getTime() - new Date(start).getTime()
    return `${(duration / 1000).toFixed(1)}s`
  }

  if (loading && runs.length === 0) {
    return (
      <div className="run-list-page">
        <div className="loading">Loading runs...</div>
      </div>
    )
  }

  return (
    <div className="run-list-page">
      <div className="page-header">
        <h2>Benchmark Runs</h2>
        <Link to="/new-run" className="btn btn-primary">
          + New Run
        </Link>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {runs.length === 0 ? (
        <div className="empty-state">
          <p>No benchmark runs found</p>
          <Link to="/new-run" className="btn btn-primary">
            Create First Run
          </Link>
        </div>
      ) : (
        <div className="runs-table-container">
          <table className="runs-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Model</th>
                <th>Engines</th>
                <th>Batch Sizes</th>
                <th>Created</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td>
                    <span className={getStatusBadgeClass(run.status)}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.model_name}</td>
                  <td>{run.engines.join(', ')}</td>
                  <td>{run.batch_sizes.join(', ')}</td>
                  <td>{formatDate(run.created_at)}</td>
                  <td>{formatDuration(run.started_at, run.completed_at)}</td>
                  <td>
                    <Link
                      to={`/runs/${run.run_id}`}
                      className="btn btn-sm btn-secondary"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loading && runs.length > 0 && (
        <div className="update-indicator">Updating...</div>
      )}
    </div>
  )
}
