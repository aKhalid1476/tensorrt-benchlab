import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { BatchStats } from '../api/client'
import './ResultsChart.css'

interface ResultsChartProps {
  results: BatchStats[]
}

function ResultsChart({ results }: ResultsChartProps) {
  return (
    <div className="charts-container">
      <div className="chart-wrapper">
        <h3>Latency (p50 / p95)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={results}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="batch_size" stroke="#94a3b8" label={{ value: 'Batch Size', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#94a3b8" label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
            <Legend />
            <Line type="monotone" dataKey="latency_p50_ms" stroke="#10b981" name="p50 Latency" strokeWidth={2} />
            <Line type="monotone" dataKey="latency_p95_ms" stroke="#f59e0b" name="p95 Latency" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-wrapper">
        <h3>Throughput</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={results}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="batch_size" stroke="#94a3b8" label={{ value: 'Batch Size', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#94a3b8" label={{ value: 'Requests/sec', angle: -90, position: 'insideLeft' }} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
            <Legend />
            <Bar dataKey="throughput_req_per_sec" fill="#667eea" name="Throughput (req/s)" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default ResultsChart
