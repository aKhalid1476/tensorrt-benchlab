import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getTelemetry, TelemetrySample } from '../api/client'
import './TelemetryChart.css'

function TelemetryChart() {
  const [samples, setSamples] = useState<TelemetrySample[]>([])
  const [deviceName, setDeviceName] = useState<string>('Unknown')

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const data = await getTelemetry()
        setSamples(data.samples)
        setDeviceName(data.device_name)
      } catch (error) {
        console.error('Failed to fetch telemetry:', error)
      }
    }

    fetchTelemetry()
    const interval = setInterval(fetchTelemetry, 3000)

    return () => clearInterval(interval)
  }, [])

  // Take last 50 samples for chart
  const chartData = samples.slice(-50).map((sample, idx) => ({
    index: idx,
    gpu_util: sample.gpu_utilization_percent,
    memory_percent: (sample.memory_used_mb / sample.memory_total_mb) * 100,
  }))

  return (
    <div className="telemetry-container">
      <div className="telemetry-header">
        <h3>{deviceName}</h3>
        {samples.length > 0 && (
          <div className="telemetry-stats">
            <span>GPU: {samples[samples.length - 1].gpu_utilization_percent.toFixed(1)}%</span>
            <span>Memory: {samples[samples.length - 1].memory_used_mb.toFixed(0)} / {samples[samples.length - 1].memory_total_mb.toFixed(0)} MB</span>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="index" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" domain={[0, 100]} label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
          <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
          <Legend />
          <Line type="monotone" dataKey="gpu_util" stroke="#10b981" name="GPU Utilization" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="memory_percent" stroke="#3b82f6" name="Memory Usage" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default TelemetryChart
