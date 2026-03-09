import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createRun, EngineType } from '../api/client'
import '../styles/NewRunPage.css'

export default function NewRunPage() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    runner_url: 'http://localhost:8001',
    model_name: 'resnet50',
    engines: ['pytorch_cpu'] as EngineType[],
    batch_sizes: '1, 4, 8, 16',
    num_iterations: 50,
    warmup_iterations: 10,
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleEngineToggle = (engine: EngineType) => {
    setFormData((prev) => ({
      ...prev,
      engines: prev.engines.includes(engine)
        ? prev.engines.filter((e) => e !== engine)
        : [...prev.engines, engine],
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    // Parse batch sizes
    const batchSizes = formData.batch_sizes
      .split(',')
      .map((s) => parseInt(s.trim()))
      .filter((n) => !isNaN(n))

    if (batchSizes.length === 0) {
      setError('Please specify at least one batch size')
      return
    }

    if (formData.engines.length === 0) {
      setError('Please select at least one engine')
      return
    }

    try {
      setSubmitting(true)
      const response = await createRun({
        runner_url: formData.runner_url,
        model_name: formData.model_name,
        engines: formData.engines,
        batch_sizes: batchSizes,
        num_iterations: formData.num_iterations,
        warmup_iterations: formData.warmup_iterations,
      })

      // Navigate to run detail page
      navigate(`/runs/${response.run_id}`)
    } catch (err) {
      setError('Failed to create run: ' + (err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="new-run-page">
      <div className="page-header">
        <h2>Create New Benchmark Run</h2>
      </div>

      <form onSubmit={handleSubmit} className="run-form">
        {error && <div className="error-banner">{error}</div>}

        <div className="form-group">
          <label>Runner URL *</label>
          <input
            type="text"
            value={formData.runner_url}
            onChange={(e) =>
              setFormData({ ...formData, runner_url: e.target.value })
            }
            placeholder="http://localhost:8001"
            required
          />
          <small>URL of the GPU runner service</small>
        </div>

        <div className="form-group">
          <label>Model *</label>
          <select
            value={formData.model_name}
            onChange={(e) =>
              setFormData({ ...formData, model_name: e.target.value })
            }
          >
            <option value="resnet50">ResNet50</option>
          </select>
        </div>

        <div className="form-group">
          <label>Engines *</label>
          <div className="checkbox-group">
            {(['pytorch_cpu', 'pytorch_cuda', 'tensorrt'] as EngineType[]).map(
              (engine) => (
                <label key={engine} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={formData.engines.includes(engine)}
                    onChange={() => handleEngineToggle(engine)}
                  />
                  {engine}
                </label>
              )
            )}
          </div>
        </div>

        <div className="form-group">
          <label>Batch Sizes *</label>
          <input
            type="text"
            value={formData.batch_sizes}
            onChange={(e) =>
              setFormData({ ...formData, batch_sizes: e.target.value })
            }
            placeholder="1, 4, 8, 16"
            required
          />
          <small>Comma-separated list of batch sizes</small>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Measurement Iterations</label>
            <input
              type="number"
              value={formData.num_iterations}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  num_iterations: parseInt(e.target.value),
                })
              }
              min="1"
              max="1000"
            />
          </div>

          <div className="form-group">
            <label>Warmup Iterations</label>
            <input
              type="number"
              value={formData.warmup_iterations}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  warmup_iterations: parseInt(e.target.value),
                })
              }
              min="0"
              max="100"
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn btn-secondary"
            disabled={submitting}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting}
          >
            {submitting ? 'Creating...' : 'Create Run'}
          </button>
        </div>
      </form>
    </div>
  )
}
