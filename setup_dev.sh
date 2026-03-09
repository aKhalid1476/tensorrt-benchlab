#!/bin/bash
# TensorRT BenchLab - Development Setup Script

set -e  # Exit on error

echo "🚀 Setting up TensorRT BenchLab for development..."

# Install contracts package (required by controller and runner)
echo ""
echo "📦 Installing contracts package..."
cd contracts
pip install -e ".[dev]"
cd ..

# Install controller
echo ""
echo "🎛️  Installing controller..."
cd controller
pip install -e ".[dev]"
cd ..

# Install runner (requires CUDA/PyTorch)
echo ""
echo "🏃 Installing runner..."
cd runner
pip install -e ".[dev]"
cd ..

# Install frontend dependencies
echo ""
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start controller: cd controller && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "  2. Start runner:     cd runner && uvicorn app.main:app --host 0.0.0.0 --port 8001"
echo "  3. Start frontend:   cd frontend && npm run dev"
