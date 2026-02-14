.PHONY: help dev build up down test lint bench clean install

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies for backend and frontend
	@echo "Installing backend dependencies..."
	cd backend && pip install -e .[dev]
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

dev: ## Run development servers (backend + frontend)
	docker-compose up

build: ## Build Docker images
	docker-compose build

up: ## Start services in detached mode
	docker-compose up -d

down: ## Stop all services
	docker-compose down

test: ## Run tests for backend and frontend
	@echo "Running backend tests..."
	cd backend && python -m pytest tests/ -v
	@echo "Running frontend tests..."
	cd frontend && npm run test

lint: ## Run linters for backend and frontend
	@echo "Linting backend..."
	cd backend && ruff check app/ && mypy app/
	@echo "Linting frontend..."
	cd frontend && npm run lint

bench: ## Run a quick benchmark (ResNet50 CPU, batch size 1)
	@echo "Running quick benchmark..."
	curl -X POST http://localhost:8000/bench/run \
		-H "Content-Type: application/json" \
		-d '{"model_name":"resnet50","engine_type":"pytorch_cpu","batch_sizes":[1],"num_iterations":10,"warmup_iterations":2}'

clean: ## Clean up generated files and caches
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/data/*.json
	@echo "Clean complete!"

logs: ## Show logs from all services
	docker-compose logs -f

status: ## Show status of all services
	docker-compose ps
