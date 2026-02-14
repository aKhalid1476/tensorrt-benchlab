import { useState } from 'react'
import { startBenchmark, BenchmarkRequest } from '../api/client'
import './BenchmarkControls.css'

interface BenchmarkControlsProps {
  onBenchmarkStart: (runId: string) => void
  isLoading: boolean
}

function BenchmarkControls({ onBenchmarkStart, isLoading }: BenchmarkControlsProps) {
  const [modelName, setModelName] = useState('resnet50')
  const [engineType, setEngineType] = useState<'pytorch_cpu' | 'pytorch_gpu' | 'tensorrt'>('pytorch_cpu')
  const [batchSizes, setBatchSizes] = useState('1,4,8,16')
  const [numIterations, setNumIterations] = useState(100)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const request: BenchmarkRequest = {
      model_name: modelName,
      engine_type: engineType,
      batch_sizes: batchSizes.split(',').map(n => parseInt(n.trim())),
      num_iterations: numIterations,
      warmup_iterations: 3,
    }

    try {
      const response = await startBenchmark(request)
      onBenchmarkStart(response.run_id)
    } catch (error) {
      console.error('Failed to start benchmark:', error)
      alert('Failed to start benchmark. Check console for details.')
    }
  }

  return (
    <form className="benchmark-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="model">Model</label>
        <select
          id="model"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          disabled={isLoading}
        >
          <option value="resnet50">ResNet50</option>
          <option value="mobilenet_v2">MobileNetV2</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="engine">Engine</label>
        <select
          id="engine"
          value={engineType}
          onChange={(e) => setEngineType(e.target.value as any)}
          disabled={isLoading}
        >
          <option value="pytorch_cpu">PyTorch CPU</option>
          <option value="pytorch_gpu">PyTorch GPU</option>
          <option value="tensorrt">TensorRT</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="batches">Batch Sizes (comma-separated)</label>
        <input
          id="batches"
          type="text"
          value={batchSizes}
          onChange={(e) => setBatchSizes(e.target.value)}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="iterations">Iterations</label>
        <input
          id="iterations"
          type="number"
          value={numIterations}
          onChange={(e) => setNumIterations(parseInt(e.target.value))}
          min="1"
          disabled={isLoading}
        />
      </div>

      <button type="submit" className="run-button" disabled={isLoading}>
        {isLoading ? 'Running...' : 'Run Benchmark'}
      </button>
    </form>
  )
}

export default BenchmarkControls
