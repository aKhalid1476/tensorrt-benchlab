import { useState } from 'react'
import BenchmarkDashboard from './pages/BenchmarkDashboard'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>⚡ TensorRT BenchLab</h1>
        <p>Production-quality inference benchmarking</p>
      </header>
      <main className="app-main">
        <BenchmarkDashboard />
      </main>
    </div>
  )
}

export default App
