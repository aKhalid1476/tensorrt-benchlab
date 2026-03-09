.PHONY: help install test lint clean dev-controller dev-runner build-runner run-runner build-controller status

help: ## Show this help message
	@echo '⚡ TensorRT BenchLab - Makefile Commands'
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ==============================================================================
# Installation
# ==============================================================================

install: ## Install all packages in editable mode
	@echo "📦 Installing contracts..."
	cd contracts && pip install -e ".[dev]"
	@echo "📦 Installing controller..."
	cd controller && pip install -e ".[dev]"
	@echo "📦 Installing runner..."
	cd runner && pip install -e ".[dev]"
	@echo "✅ Installation complete!"

install-contracts: ## Install only contracts package
	cd contracts && pip install -e ".[dev]"

install-controller: ## Install controller (requires contracts)
	cd contracts && pip install -e .
	cd controller && pip install -e ".[dev]"

install-runner: ## Install runner (requires contracts)
	cd contracts && pip install -e .
	cd runner && pip install -e ".[dev]"

# ==============================================================================
# Testing
# ==============================================================================

test: ## Run all tests
	@echo "🧪 Running contract tests..."
	cd contracts && pytest -v
	@echo "🧪 Running controller tests..."
	cd controller && pytest -v || echo "⚠️  Controller tests not implemented yet"
	@echo "🧪 Running runner tests..."
	cd runner && pytest -v || echo "⚠️  Runner tests not implemented yet"

test-contracts: ## Test contracts package
	cd contracts && pytest -v

test-controller: ## Test controller
	cd controller && pytest -v

test-runner: ## Test runner
	cd runner && pytest -v

# ==============================================================================
# Linting
# ==============================================================================

lint: ## Run linters on all packages
	@echo "🔍 Linting contracts..."
	cd contracts && ruff check contracts/ || true
	@echo "🔍 Linting controller..."
	cd controller && ruff check app/ || true
	@echo "🔍 Linting runner..."
	cd runner && ruff check app/ || true

# ==============================================================================
# Development Servers
# ==============================================================================

dev-controller: ## Run controller locally (port 8000)
	@echo "🚀 Starting controller on http://localhost:8000"
	cd controller && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-runner: ## Run runner locally (port 8001, requires GPU)
	@echo "🚀 Starting runner on http://localhost:8001"
	cd runner && uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# ==============================================================================
# Docker - Runner
# ==============================================================================

build-runner: ## Build runner Docker image
	@echo "🐋 Building runner Docker image..."
	docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
	@echo "✅ Runner image built: tensorrt-benchlab-runner"

run-runner: ## Run runner in Docker with GPU (--gpus all)
	@echo "🐋 Starting runner container with GPU access..."
	docker run -d --name tensorrt-benchlab-runner \
		--gpus all \
		-p 8001:8001 \
		-v $$(pwd)/cache:/app/cache \
		-e BENCHLAB_LOG_LEVEL=INFO \
		tensorrt-benchlab-runner
	@echo "✅ Runner started on http://localhost:8001"
	@echo "🔍 Check logs: docker logs -f tensorrt-benchlab-runner"

stop-runner: ## Stop runner container
	docker stop tensorrt-benchlab-runner || true
	docker rm tensorrt-benchlab-runner || true

logs-runner: ## Show runner container logs
	docker logs -f tensorrt-benchlab-runner

# ==============================================================================
# Docker - Controller + Frontend
# ==============================================================================

build-controller: ## Build controller Docker image
	@echo "🐋 Building controller Docker image..."
	docker-compose build controller
	@echo "✅ Controller image built"

up: ## Start controller (and frontend if available) via docker-compose
	@echo "🐋 Starting services..."
	docker-compose up -d
	@echo "✅ Services started"
	@echo "📍 Controller: http://localhost:8000"
	@echo "📍 API Docs: http://localhost:8000/docs"

down: ## Stop all docker-compose services
	docker-compose down

logs: ## Show logs from all docker-compose services
	docker-compose logs -f

status: ## Show status of all services
	docker-compose ps

# ==============================================================================
# API Testing
# ==============================================================================

test-runner-health: ## Test runner health endpoint
	@echo "🏥 Testing runner health..."
	curl -f http://localhost:8001/health && echo "" || echo "❌ Runner not healthy"

test-runner-version: ## Test runner version endpoint
	@echo "📋 Runner version info:"
	curl -s http://localhost:8001/version | jq .

test-controller-health: ## Test controller health endpoint
	@echo "🏥 Testing controller health..."
	curl -f http://localhost:8000/health && echo "" || echo "❌ Controller not healthy"

bench: ## Run a quick benchmark via controller
	@echo "⚡ Creating benchmark run..."
	@curl -s -X POST http://localhost:8000/runs \
		-H "Content-Type: application/json" \
		-d '{"runner_url":"http://localhost:8001","model_name":"resnet50","engines":["pytorch_cpu"],"batch_sizes":[1,4],"num_iterations":20}' | jq .

# ==============================================================================
# Cleanup
# ==============================================================================

clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf cache/*.onnx cache/*.trt cache/*.json
	rm -rf data/controller/*.db
	@echo "✅ Clean complete!"

clean-docker: ## Remove all Docker containers and images
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker rmi tensorrt-benchlab-runner tensorrt-benchlab-controller 2>/dev/null || true
	@echo "✅ Docker clean complete!"

# ==============================================================================
# Development Workflow
# ==============================================================================

dev: ## Full local development setup (install + controller + instructions)
	@echo "🚀 Setting up local development environment..."
	@echo ""
	@echo "Step 1: Installing packages..."
	@make install
	@echo ""
	@echo "Step 2: Starting controller..."
	@echo ""
	@echo "📍 Run these commands in separate terminals:"
	@echo ""
	@echo "  Terminal 1 (Controller):"
	@echo "    make dev-controller"
	@echo ""
	@echo "  Terminal 2 (Runner - requires GPU):"
	@echo "    make dev-runner"
	@echo ""
	@echo "Then test with:"
	@echo "  make bench"
