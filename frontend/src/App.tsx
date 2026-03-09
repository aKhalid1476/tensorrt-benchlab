import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import RunListPage from './pages/RunListPage'
import NewRunPage from './pages/NewRunPage'
import RunDetailPage from './pages/RunDetailPage'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <Link to="/" className="logo">
            <h1>⚡ TensorRT BenchLab</h1>
          </Link>
          <p>Production-Quality Inference Benchmarking</p>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<RunListPage />} />
            <Route path="/new-run" element={<NewRunPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
